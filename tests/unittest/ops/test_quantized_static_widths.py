# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.
import os
import tempfile
import unittest

import torch
from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import IntVar, Tensor
from aitemplate.testing import detect_target

# WGMMA kernels require the architecture-specific SM90a feature set.
os.environ.setdefault("AIT_FORCE_CUTLASS_SM90_KERNELS", "1")


class QuantizedStaticWidthsTest(unittest.TestCase):
    @torch.no_grad()
    def test_fp8_partial_rows_and_zero_scale(self):
        target = detect_target(use_fp16_acc=False)
        if target.name() != 'cuda' or target._arch not in ('90', '100'):
            self.skipTest('SM90/SM100 quantized kernels')
        rows = IntVar([1, 257], name='rows')
        outputs = []
        widths = (8, 192, 384, 576, 1152)
        for i, c in enumerate(widths):
            x = Tensor([rows, c], name=f'x{i}', is_input=True)
            g = Tensor([c], name=f'g{i}', is_input=True)
            z = Tensor([rows, 2*c], name=f'z{i}', is_input=True)
            outputs.extend(ops.quantize_to_fp8()(x))
            outputs.extend(ops.rms_quantize()(x))
            outputs.extend(ops.rmsnorm(fp8_out=True)(x, g))
            outputs.extend(ops.swiglu(fp8_out=True)(z))
        for i, output in enumerate(outputs):
            output._attrs.update(name=f'out{i}', is_output=True)
        with tempfile.TemporaryDirectory(prefix='ait_fp8_widths_') as work:
            model = compile_model(outputs, target, work, 'model')
            for n in (1, 3, 7, 257):
                torch.manual_seed(n)
                inputs, refs, buffers = {}, [], []
                for i, c in enumerate(widths):
                    x = torch.randn(n, c, device='cuda', dtype=torch.float16)
                    x[0].zero_()
                    g = torch.randn(c, device='cuda', dtype=torch.float16)
                    z = torch.randn(n, 2*c, device='cuda', dtype=torch.float16)
                    z[0].zero_()
                    inputs.update({f'x{i}': x, f'g{i}': g, f'z{i}': z})
                    rrms = torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
                    norm = x.float() * rrms * g.float()
                    gate, up = z.float().chunk(2, -1)
                    swiglu = torch.nn.functional.silu(gate) * up
                    for j, value in enumerate((x.float(), x.float(), norm, swiglu)):
                        scale = value.abs().amax(-1, keepdim=True) / 448
                        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
                        # Producers store intermediate values in FP16 before FP8 conversion.
                        q = (value.half().float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
                        refs.extend([q, scale])
                        buffers.extend([torch.empty_like(q), torch.empty_like(scale)])
                        if j == 1:
                            refs.append(rrms.half()); buffers.append(torch.empty_like(rrms.half()))
                model.run_with_tensors(inputs, buffers)
                for got, expected in zip(buffers, refs):
                    # Quantization boundaries can differ by one FP8 step from FP32 reduction order.
                    if got.dtype == torch.float8_e4m3fn:
                        torch.testing.assert_close(got.float(), expected.float(), atol=0.5, rtol=0.125)
                        self.assertTrue(torch.equal(got[0].float(), expected[0].float()))
                    else:
                        torch.testing.assert_close(got, expected, atol=0.002, rtol=0.002)

    @torch.no_grad()
    def test_fp8_graph_replay_after_batch_growth(self):
        target = detect_target(use_fp16_acc=False)
        if target.name() != 'cuda' or target._arch not in ('90', '100'):
            self.skipTest('SM90/SM100 FP8 GEMM')
        rows = IntVar([1, 8193], name='rows')
        x = Tensor([rows, 192], name='x', dtype='float8_e4m3', is_input=True)
        w = Tensor([192, 192], name='w', dtype='float8_e4m3', is_input=True)
        sx = Tensor([rows, 1], name='sx', dtype='float32', is_input=True)
        sw = Tensor([1], name='sw', dtype='float32', is_input=True)
        residual = Tensor([rows, 192], name='residual', is_input=True)
        out = ops.gemm_rcr_fp8_fused()(x, w, sx, sw, residual)
        out._attrs.update(name='out', is_output=True)
        plain = ops.gemm_rcr_fp8_fused()(x, w, sx, sw)
        plain._attrs.update(name='plain', is_output=True)
        with tempfile.TemporaryDirectory(prefix='ait_fp8_replay_') as work:
            model = compile_model([out, plain], target, work, 'model')
            saved = {}
            for n in (1, 3, 5632, 5633, 6144, 6145, 8191, 8192, 8193, 1, 3, 5632, 5633, 6144, 8191, 8192):
                if n not in saved:
                    torch.manual_seed(n)
                    inputs = dict(
                        x=torch.randn(n, 192, device='cuda').to(torch.float8_e4m3fn),
                        w=torch.ones(192, 192, device='cuda').to(torch.float8_e4m3fn),
                        sx=torch.rand(n, 1, device='cuda'),
                        sw=torch.full((1,), 0.0 if n == 3 else 0.37, device='cuda'),
                        residual=torch.randn(n, 192, device='cuda', dtype=torch.float16))
                    actual = torch.empty(n, 192, device='cuda', dtype=torch.float16)
                    inputs['sx'][0].zero_()
                    scaled = inputs['x'].float().sum(-1, keepdim=True) * (inputs['sx'] * inputs['sw'])
                    expected = (scaled + inputs['residual'].float()).half()
                    plain_expected = scaled.expand(n, 192).contiguous().half()
                    plain_actual = torch.empty_like(actual)
                    saved[n] = inputs, actual, expected, plain_actual, plain_expected
                    model.run_with_tensors(inputs, [actual, plain_actual], graph_mode=False)
                inputs, actual, expected, plain_actual, plain_expected = saved[n]
                # Reuse the original tensor addresses to exercise the cached graph.
                model.run_with_tensors(inputs, [actual, plain_actual], graph_mode=True)
                torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.002)
                torch.testing.assert_close(plain_actual, plain_expected, atol=0.002, rtol=0.002)

    @torch.no_grad()
    def test_mxfp8_lane_groups_and_dispatch(self):
        target = detect_target(use_fp16_acc=False)
        if target.name() != 'cuda' or target._arch != '100':
            self.skipTest('SM100 MXFP8')
        cases = [(32, 'none'), (192, 'rms_out'), (384, 'rmsnorm'),
                 (576, 'swiglu'), (1152, 'rms_out')]
        rows = IntVar([1, 10369], name='rows')
        outputs = []
        for i, (k, mode) in enumerate(cases):
            x = Tensor([rows, k * (2 if mode == 'swiglu' else 1)], name=f'x{i}', is_input=True)
            w = Tensor([192, k], name=f'w{i}', dtype='float8_e4m3', is_input=True)
            sf = Tensor([2 * ((k//32+3)//4) * 512], name=f'sf{i}', dtype='float8_e4m3', is_input=True)
            kwargs = {}
            if mode == 'rmsnorm':
                kwargs['gamma'] = Tensor([k], name=f'g{i}', is_input=True)
            if mode == 'swiglu':
                kwargs['rrms'] = Tensor([rows, 1], name=f'r{i}', is_input=True)
            result = ops.gemm_rcr_mxfp8()(x, w, sf, norm=mode, **kwargs)
            outputs.extend(result if isinstance(result, tuple) else [result])
        for i, output in enumerate(outputs):
            output._attrs.update(name=f'out{i}', is_output=True)
        with tempfile.TemporaryDirectory(prefix='ait_mxfp8_groups_') as work:
            model = compile_model(outputs, target, work, 'model')
            for n in (1, 3, 7, 648, 649, 6144, 6145, 8191, 8192, 8193, 10368, 10369):
                torch.manual_seed(n)
                inputs, expected = {}, []
                for i, (k, mode) in enumerate(cases):
                    # Exact binary fractions make sum-of-squares independent of the
                    # reduction order, avoiding FP8 tie crossings in the oracle.
                    x = (torch.randint(-4, 5, (n, k * (2 if mode == 'swiglu' else 1)), device='cuda').float() / 16).half()
                    x[0].zero_()
                    inputs[f'x{i}'] = x
                    inputs[f'w{i}'] = torch.ones(192, k, device='cuda').to(torch.float8_e4m3fn)
                    inputs[f'sf{i}'] = torch.full((2*((k//32+3)//4)*512,), 127, device='cuda', dtype=torch.uint8).view(torch.float8_e4m3fn)
                    value = x.float()
                    rrms = torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
                    if mode == 'rmsnorm':
                        inputs[f'g{i}'] = torch.rand(k, device='cuda', dtype=torch.float16)
                        value = value * rrms * inputs[f'g{i}'].float()
                    if mode == 'swiglu':
                        inputs[f'r{i}'] = torch.rand(n, 1, device='cuda', dtype=torch.float16)
                        gate, up = (value * inputs[f'r{i}'].float()).chunk(2, -1)
                        value = torch.nn.functional.silu(gate) * up
                    amax = value.reshape(n, -1, 32).abs().amax(-1, keepdim=True)
                    scale = torch.where(amax > 0, 2.0 ** torch.ceil(torch.log2(amax / 448)), torch.ones_like(amax))
                    # CUDA FP8 conversions saturate; PyTorch casts otherwise emit NaN.
                    q = (value.half().float().reshape(n, -1, 32)/scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale
                    expected.append(q.reshape(n, k).sum(-1, keepdim=True).expand(n, 192).contiguous().half())
                    if mode == 'rms_out':
                        expected.append(rrms.half())
                buffers = [torch.empty_like(v) for v in expected]
                # Warm up once only: entering a new dispatch bucket must not
                # allocate memory inside graph capture.
                if n == 1:
                    model.run_with_tensors(inputs, buffers, graph_mode=False)
                model.run_with_tensors(inputs, buffers, graph_mode=True)
                for index, (got, ref) in enumerate(zip(buffers, expected)):
                    with self.subTest(rows=n, output=index):
                        torch.testing.assert_close(got, ref, atol=0.01, rtol=0.002)


if __name__ == '__main__':
    unittest.main()
