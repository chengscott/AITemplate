"""Graph-replay profiling and code generation for native projection fusions."""

import hashlib
import json
import math
import logging
import os
from pathlib import Path
import subprocess
import statistics

from aitemplate.backend.target import Target
from aitemplate.compiler.base import ExecItem
from aitemplate.utils import environ

_LOGGER = logging.getLogger(__name__)


def candidates(op, arch, joint=True):
    if arch == 80 and op == "gemm_rms_swiglu":
        result = {"fallback": None}
        for tm, tn, stages in (
            (32, 32, 3),
            (32, 64, 3),
            (64, 64, 3),
            (128, 64, 3),
            (64, 128, 3),
            (128, 128, 3),
            (64, 64, 4),
            (128, 64, 4),
        ):
            for inline in (False, True):
                result[f"m{tm}n{tn}s{stages}i{int(inline)}"] = dict(
                    tm=tm, tn=tn, stages=stages, inline=inline
                )
        return result
    if arch not in (90, 100):
        raise NotImplementedError(f"Native fusion profiling: {op} on SM{arch}")
    result = {"fallback": None}
    clusters = [(0, 0), (1, 0), (0, 1), (1, 1)] if arch == 90 else [(0, 0)]
    if op == "gemm_rms_relu_gemm":
        schedules = (
            [(64, False), (128, False), (128, True)] if arch == 90 else [(128, False)]
        )
        for tm, pp in schedules:
            for tn in (64, 128, 192):
                for pc, cc in clusters:
                    buffers = (
                        [(False, False), (False, True), (True, False), (True, True)]
                        if arch == 100
                        else [(pp, pp)]
                    )
                    for pi, ci in buffers:
                        suffix = f"i{int(pi)}{int(ci)}" if arch == 100 else ""
                        result[f"m{tm}n{tn}p{int(pp)}c{pc}{cc}{suffix}"] = dict(
                            tm=tm, tn=tn, pp=pp, pc=pc, cc=cc, pi=pi, ci=ci
                        )
    elif op == "gemm_rms_swiglu":
        if not joint:
            clusters = [(pc, cc) for pc, cc in clusters if pc == 0]
        tiles = (
            [(64, n, False, False) for n in (64, 128, 192)]
            + [(128, 64, True, True), (128, 128, True, True), (128, 192, True, False)]
            if arch == 90
            else [(128, n, False, False) for n in (64, 128)]
        )
        for tm, tn, pp, early in tiles:
            for pc, cc in clusters:
                result[f"m{tm}n{tn}p{int(pp)}e{int(early)}c{pc}{cc}"] = dict(
                    tm=tm, tn=tn, pp=pp, early=early, pc=pc, cc=cc
                )
    elif op == "gemm_mxfp8_swiglu" and arch == 100:
        for tn in (64, 128, 256):
            for dn in (64, 128, 192):
                result[f"n{tn}d{dn}"] = dict(tn=tn, dn=dn)
    else:
        raise NotImplementedError(f"Native fusion profiling: {op} on SM{arch}")
    return result


def call_body(attrs, arch, config, eps=None):
    """Return an integer-status C++ body; pointers follow the operator signature."""
    op = attrs["op"]
    eps = eps if eps is not None else f"{attrs['eps']}f"
    joint = str(attrs.get("joint", False)).lower()
    if op == "gemm_rms_relu_gemm":
        args = f"x,w,skip,gamma,up,outer,out,workspace,int(rows),{eps},stream"
        if config is not None:
            c = config
            return f"return ait::native_fusion::boundary::run<{arch},{c['tm']},{str(c['pp']).lower()},{c['tn']},{str(c['pi']).lower()},{str(c['ci']).lower()}>({args},{c['pc']},{c['cc']});"
        small = 64 if arch == 90 else 128
        pp = str(arch == 90).lower()
        return f"if(rows<=2592) return ait::native_fusion::boundary::run<{arch},{small},false,64>({args});\nif(rows<=10368) return ait::native_fusion::boundary::run<{arch},128,false,128>({args});\nreturn ait::native_fusion::boundary::run<{arch},128,{pp},192>({args});"
    if op == "gemm_rms_swiglu":
        args = f"x,w,skip,proj,updated,out,workspace,int(rows),{eps},stream"
        if arch == 80:
            c = config or dict(tm=64, tn=64, stages=3, inline=False)
            return f"return ait::native_fusion::sm80::run<{c['tm']},{c['tn']},{c['stages']},{joint},{str(c['inline']).lower()}>({args});"
        if config is None:
            return f"return ait::native_fusion::swiglu::run<{arch},{joint}>({args});"
        c = config
        return f"return ait::native_fusion::swiglu::run_tuned<{arch},{joint},{c['tm']},{c['tn']},{str(c['pp']).lower()},{str(c['early']).lower()}>({args},{c['pc']},{c['cc']});"
    args = f"x,w,sf,down,down_sf,out,workspace,int(rows),{eps},stream"
    if config is not None:
        return f"return ait::native_fusion::mxfp8::run<{config['tn']},{config['dn']}>({args});"
    return f"if(rows<=648) return ait::native_fusion::mxfp8::run<64>({args});\nif(rows<=10368) return ait::native_fusion::mxfp8::run<128>({args});\nreturn ait::native_fusion::mxfp8::run<256>({args});"


def function(attrs, arch, signature):
    op = attrs["op"]
    header = {
        "gemm_rms_relu_gemm": "native_rms_relu_gemm.h",
        "gemm_rms_swiglu": "native_rms_swiglu.h",
        "gemm_mxfp8_swiglu": "native_mxfp8_swiglu.h",
    }[op]
    if arch == 80:
        header = "native_rms_swiglu_sm80.h"
    body = []
    choices = candidates(op, arch, attrs.get("joint", True))
    for row, item in sorted(
        attrs.get("exec_path", {}).items(), key=lambda x: int(x[0])
    ):
        if item.algo and item.algo != "fallback":
            if item.algo not in choices:
                raise ValueError(f"Unknown native fusion configuration {item.algo}")
            body.append(
                f"if(rows=={int(row)}) {{ {call_body(attrs,arch,choices[item.algo])} }}"
            )
    body.append(call_body(attrs, arch, None))
    attrs.pop("cutedsl_obj_path", None)
    return (
        f'#include "{header}"\n#include <stdexcept>\n{signature} {{\nint status = [&]() -> int {{\n'
        + "\n".join(body)
        + '\n}();\nif(status) throw std::runtime_error("Native fused GEMM launch failed");\n}\n'
    )


def profile_rows(attrs):
    dims = attrs["inputs"][0].shape()[:-1]
    lo = math.prod(d.lower_bound() for d in dims)
    hi = math.prod(d.upper_bound() for d in dims)
    requested = attrs.get("profile_rows")
    if requested is not None:
        if any(type(m) is not int or m < lo or m > hi for m in requested):
            raise ValueError(f"profile_rows must contain integers in [{lo}, {hi}]")
        return sorted(set(requested))
    dynamic = [d for d in dims if d.lower_bound() != d.upper_bound()]
    if not dynamic:
        return [hi]
    if len(dynamic) != 1:
        # Exact specializations only: no inference about products of dynamic dims.
        return sorted({lo, hi})
    d = dynamic[0]
    scale = hi // d.upper_bound()
    values = set(d._attrs["values"])
    power = 1
    while power <= d.upper_bound():
        if power >= d.lower_bound():
            values.add(power)
        power *= 2
    return sorted(v * scale for v in values)


def _native_enabled():
    from .native_fused_gemm import enabled, target_arch

    default = "1" if target_arch() == 80 else "0"
    return enabled() and os.environ.get("AIT_NATIVE_FUSION_TUNING", default) == "1"


def profile_context(attrs):
    """Recognize a connected residual/SwiGLU/boundary pair with matching skip."""
    op = attrs["op"]
    if op == "gemm_rms_relu_gemm":
        sources = attrs["inputs"][0].src_ops()
        if len(sources) != 1:
            return None
        ffn = next(iter(sources))._attrs
        boundary = attrs
    elif op == "gemm_rms_swiglu" and attrs.get("joint"):
        outputs = attrs.get("outputs", [])
        if len(outputs) != 2 or len(outputs[1].dst_ops()) != 1:
            return None
        boundary = next(iter(outputs[1].dst_ops()))._attrs
        ffn = attrs
    else:
        return None
    if (
        ffn["op"] != "gemm_rms_swiglu"
        or not ffn.get("joint")
        or boundary["op"] != "gemm_rms_relu_gemm"
        or boundary["inputs"][0] is not ffn["outputs"][1]
        or boundary["inputs"][2] is not ffn["outputs"][0]
    ):
        return None
    return {
        "kind": "swiglu_boundary",
        "ffn_eps": ffn["eps"],
        "boundary_eps": boundary["eps"],
    }


def gen_profiler(attrs, workdir):
    attrs["exec_path"] = {}
    attrs.pop("native_profile_results", None)
    if not _native_enabled() or Target.current().use_dummy_profiling_results():
        return []
    arch = int(Target.current()._arch)
    options = candidates(attrs["op"], arch, attrs.get("joint", True))
    rows = profile_rows(attrs)
    if not rows:
        return []
    attrs["exec_path"] = {
        str(m): ExecItem(profiling_key=str(m), exec_cond=f"rows == {m}", algo="")
        for m in rows
    }
    from .native_fusion_profiler_source import render

    attrs["native_profile_context"] = profile_context(attrs) if arch != 80 else None
    source = render(attrs, arch, options)
    # Include header contents in the generated-source cache identity.
    kernels = Path(Target.current().static_files_path) / "include/kernels"
    digest = hashlib.sha256(source.encode())
    digest.update(Path(__file__).read_bytes())
    digest.update(Target.current().compile_options().encode())
    roots = [
        ("native", kernels, list(kernels.glob("native_*.h"))),
    ]
    template = Path(Target.current().template_path())
    for name, root in (
        ("cutlass", template / "include"),
        ("cutlass_util", template / "tools/util/include"),
        ("cutlass_dual_gemm", template / "examples/45_dual_gemm"),
    ):
        roots.append(
            (
                name,
                root,
                [
                    p
                    for p in root.rglob("*")
                    if p.suffix in (".h", ".hpp", ".inl", ".cuh")
                ],
            )
        )
    for name, root, headers in roots:
        for p in sorted(headers):
            digest.update(f"{name}/{p.relative_to(root)}".encode())
            digest.update(p.read_bytes())
    fingerprint = digest.hexdigest()
    source += f"\n// Native header fingerprint: {fingerprint}\n"
    # The builder invokes make from the profiler directory, not the caller's cwd.
    directory = Path(workdir).resolve() / "profiler" / "native_fusion"
    directory.mkdir(parents=True, exist_ok=True)
    prefix = directory / fingerprint
    path = prefix.with_suffix(".cu")
    path.write_text(source)
    attrs["native_profiler"] = str(prefix)
    attrs["native_profile_fingerprint"] = fingerprint
    return [(str(path), str(prefix))]


def profile(attrs, devices):
    if not attrs.get("exec_path") or not _native_enabled():
        return
    target = Target.current()
    executable = attrs["native_profiler"]
    env = os.environ.copy()
    env[target.dev_select_flag()] = str((devices or [0])[0])
    timeout = int(os.environ.get("AIT_PROFILER_TIMEOUT", "500"))

    def invoke(*args):
        try:
            result = subprocess.run(
                [executable, *map(str, args)],
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=True,
            )
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                f"Native fusion profiler failed: {error.stderr[-4096:]}"
            ) from error
        return json.loads(result.stdout)

    identity = invoke("--info")
    if identity["arch"] != int(target._arch):
        raise RuntimeError(f"Native profiler GPU does not match target: {identity}")
    fingerprint = attrs["native_profile_fingerprint"]
    options = candidates(attrs["op"], int(target._arch), attrs.get("joint", True))
    attrs["native_profile_results"] = {}
    for row, item in attrs["exec_path"].items():
        specification = dict(
            op=attrs["op"],
            rows=int(row),
            eps=attrs["eps"],
            joint=attrs.get("joint", False),
            device=identity,
            context=attrs.get("native_profile_context"),
            source=fingerprint,
        )
        serialized = json.dumps(specification, sort_keys=True)
        key = hashlib.sha256(serialized.encode()).hexdigest()
        cached = (
            None
            if target.force_profile()
            else target.query_profile_cache("native_fusion", {"key": key})
        )
        if cached is not None and cached.get("algo") not in options:
            cached = None
        cache_hit = cached is not None
        if cached is None:
            if environ.force_profiler_cache():
                raise RuntimeError(
                    f"Missing native fusion profile cache entry: {serialized}"
                )
            result = invoke(row, attrs["eps"])
            valid = [r for r in result["candidates"] if r["valid"]]
            if not valid or not valid[0]["id"] == 0:
                raise RuntimeError("Native fusion fallback failed numerical validation")
            # Require a margin over timing variation before replacing the fallback.
            base = valid[0]
            best = min(valid, key=lambda r: r["median_ms"])
            noise = 2 * math.sqrt(
                statistics.variance(base["capture_ms"]) / len(base["capture_ms"])
                + statistics.variance(best["capture_ms"]) / len(best["capture_ms"])
            )
            if base["median_ms"] - best["median_ms"] <= max(
                base["median_ms"] * 0.005, noise
            ):
                best = base
            cached = dict(
                algo=list(options)[best["id"]],
                measurements=result,
                specification=specification,
            )
            target.insert_profile_cache("native_fusion", {"key": key, "value": cached})
        item.algo = cached["algo"]
        _LOGGER.info(
            "Native fusion %s rows=%s: %s (%s)",
            attrs["op"],
            row,
            item.algo,
            "cache" if cache_hit else "measured",
        )
        attrs["native_profile_results"][row] = cached
