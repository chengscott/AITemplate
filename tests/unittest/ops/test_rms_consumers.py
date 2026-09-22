"""Check fused RMS consumers against the existing separate CUDA operators."""
import tempfile
import unittest

import torch
from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import Tensor, IntVar
from aitemplate.testing import detect_target


class RMSConsumersTest(unittest.TestCase):
    def test_dynamic_rows(self):
        torch.manual_seed(923)
        batch = IntVar([1, 17], name="batch")
        for channels, ffn, heads in ((192, 576, 12), (384, 256, 24)):
            shapes = {
                "source": [batch, 81, channels],
                "gateup": [batch, 81, 2 * ffn],
                "qkv": [batch, 81, 3 * heads * 16],
                "cos": [1, 81, heads, 8],
                "sin": [1, 81, heads, 8],
            }
            inputs = {k: Tensor(s, name=k, dtype="float16", is_input=True) for k, s in shapes.items()}
            x, g, qkv, co, si = [inputs[k] for k in shapes]
            rrms = ops.rms_reduce(eps=1e-6)(x)
            reference = [ops.swiglu()(g, rrms), *ops.unpack_rope(heads)(qkv, co, si, rrms)]
            fused = [ops.rms_swiglu()(g, x), *ops.rms_unpack_rope(heads)(qkv, co, si, x)]
            for i, y in enumerate(reference + fused):
                y._attrs.update(name=f"out{i}", is_output=True)
            with tempfile.TemporaryDirectory(prefix="ait-rms-consumers-") as work:
                module = compile_model(reference + fused, detect_target(), work, "test")
                for b in (1, 3, 17):
                    # This deployment runtime caches graphs by shape and requires
                    # stable pointers. Change contents in place between replays.
                    values = {k: torch.empty([b if d is batch else d for d in s], device="cuda", dtype=torch.float16)
                              for k, s in shapes.items()}
                    outputs = [torch.empty([b,81,ffn],device="cuda",dtype=torch.float16)]
                    outputs += [torch.empty([b,81,heads,16],device="cuda",dtype=torch.float16) for _ in range(3)]
                    outputs += [torch.empty_like(t) for t in outputs]
                    for magnitude in (0., 1., 8.):
                        for value in values.values():
                            value.normal_()
                        values["source"].mul_(magnitude)
                        # Keep zero-source projections finite when rrms approaches 1000.
                        if magnitude == 0:
                            values["gateup"].mul_(0.001)
                            values["qkv"].mul_(0.001)
                        for graph_mode in (False, True):
                            module.run_with_tensors(values, {f"out{i}": t for i,t in enumerate(outputs)}, graph_mode=graph_mode, sync=True)
                            for i, (ref, actual) in enumerate(zip(outputs[:4], outputs[4:])):
                                torch.testing.assert_close(actual, ref, atol=0, rtol=0,
                                    msg=lambda m: f"C={channels}, B={b}, magnitude={magnitude}, graph={graph_mode}, output={i}: {m}")

    @torch.no_grad()
    def test_large_swiglu_offsets(self):
        # First row whose packed projection offset exceeds signed int32.
        rows = (1 << 31) // 1152 + 2
        if torch.cuda.mem_get_info()[0] < 12 * (1 << 30):
            self.skipTest("Large-offset regression needs 12 GiB of free GPU memory")
        source = Tensor([rows, 192], name="source", is_input=True)
        projected = Tensor([rows, 1152], name="projected", is_input=True)
        output = ops.rms_swiglu()(projected, source)
        output._attrs.update(name="output", is_output=True)
        with tempfile.TemporaryDirectory(prefix="ait-rms-large-") as work:
            model = compile_model(output, detect_target(), work, "test")
            x = torch.full((rows, 192), 0.25, device="cuda", dtype=torch.float16)
            p = torch.full((rows, 1152), 0.125, device="cuda", dtype=torch.float16)
            actual = torch.empty((rows, 576), device="cuda", dtype=torch.float16)
            model.run_with_tensors({"source": x, "projected": p},
                                   {"output": actual}, sync=True)
            rrms = torch.rsqrt(torch.tensor(0.25 ** 2 + 1e-6, device="cuda")).half().float()
            value = 0.125 * rrms
            expected = (torch.nn.functional.silu(value) * value).half()
            for sample in (actual[:1], actual[-2:]):
                torch.testing.assert_close(sample, expected.expand_as(sample), atol=0.0002, rtol=0)

    def test_invalid_source(self):
        x = Tensor([1,81,190],dtype="float16")
        y = Tensor([1,81,1152],dtype="float16")
        with self.assertRaisesRegex(ValueError, "divisible by eight"):
            ops.rms_swiglu()(y,x)


if __name__ == "__main__":
    unittest.main()
