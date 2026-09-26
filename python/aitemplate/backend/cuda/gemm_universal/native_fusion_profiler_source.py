"""CUDA graph profiler generation for native operators and connected pairs."""

from .native_fused_gemm_profiler import call_body


def render(attrs, arch, options):
    if attrs.get("native_profile_context"):
        return _render_pair(attrs, arch, options)
    op = attrs["op"]
    if op == "gemm_rms_relu_gemm":
        header = "native_rms_relu_gemm.h"
        names = ["x", "w", "skip", "gamma", "up", "outer", "out", "workspace"]
        specs = [
            ("rows*576", 0),
            ("192*576", 0),
            ("rows*192", 0),
            ("192", 0),
            ("384*192", 0),
            ("rows*384", 0),
            ("rows*384", 0),
            ("rows*384+((rows+127)/128)*128*4+255", 3),
        ]
        outputs = [6]
    elif op == "gemm_rms_swiglu":
        header = "native_rms_swiglu_sm80.h" if arch == 80 else "native_rms_swiglu.h"
        names = ["x", "w", "skip", "proj", "updated", "out", "workspace"]
        specs = [
            ("rows*192", 0),
            ("1152*192", 0),
            ("rows*192", 0),
            ("192*192", 0),
            ("rows*192", 0),
            ("rows*576", 0),
            ("((rows+127)/128)*128*4", 3),
        ]
        outputs = [4, 5] if attrs["joint"] else [5]
    else:
        header = "native_mxfp8_swiglu.h"
        names = ["x", "w", "sf", "down", "down_sf", "out", "workspace"]
        specs = [
            ("rows*192", 0),
            ("1152*192", 1),
            ("9216", 2),
            ("192*576", 1),
            ("5120", 2),
            ("rows*192", 0),
            ("rows*770+((rows+127)/128)*3584+5*255", 3),
        ]
        outputs = [5]
    code = f'#include "{header}"\n#include "native_fusion_profiler.h"\nusing namespace ait::native_fusion::profiling;\n'
    code += (
        "std::vector<BufferSpec> specs(int64_t rows) { return {"
        + ",".join("{" + n + "," + str(t) + "}" for n, t in specs)
        + "}; }\n"
    )
    code += (
        "std::vector<int> output_indices() { return {"
        + ",".join(map(str, outputs))
        + "}; }\n"
    )
    code += "int candidate(int id, std::vector<void*> const& p, int rows, float eps, cudaStream_t stream) {\n"
    for i, name in enumerate(names):
        code += f"auto {name} = static_cast<{'uint8_t*' if name=='workspace' else 'void*'}>(p[{i}]);\n"
    code += "switch(id) {\n"
    for i, c in enumerate(options.values()):
        code += f'case {i}: {{ {call_body(attrs,arch,c,eps="eps")} }}\n'
    code += "default: return -1;\n}\n}\n"
    code += f"int main(int argc,char** argv) {{ return profile_main(argc,argv,{len(options)},specs,output_indices(),candidate); }}\n"
    return code


def _render_pair(attrs, arch, options):
    """Time the connected pair, preserving its internal PDL and buffer aliases."""
    context = attrs["native_profile_context"]
    specs = [
        ("rows*192", 0),
        ("1152*192", 4),
        ("rows*192", 0),
        ("192*192", 4),
        ("rows*192", 0),
        ("rows*576", 0),
        ("((rows+127)/128)*128*4", 3),
        ("192*576", 4),
        ("192", 0),
        ("384*192", 4),
        ("rows*384", 0),
        ("rows*384", 0),
        ("rows*384+((rows+127)/128)*128*4+255", 3),
    ]
    code = '#include "native_rms_swiglu.h"\n#include "native_fusion_profiler.h"\nusing namespace ait::native_fusion::profiling;\n'
    code += (
        "std::vector<BufferSpec> specs(int64_t rows) { return {"
        + ",".join("{" + n + "," + str(t) + "}" for n, t in specs)
        + "}; }\n"
    )
    code += "int candidate(int id, std::vector<void*> const& p, int rows, float eps, cudaStream_t stream) { switch(id) {\n"
    stages = [
        (
            "gemm_rms_swiglu",
            context["ffn_eps"],
            ["x", "w", "skip", "proj", "updated", "out", "workspace"],
            [0, 1, 2, 3, 4, 5, 6],
        ),
        (
            "gemm_rms_relu_gemm",
            context["boundary_eps"],
            ["x", "w", "skip", "gamma", "up", "outer", "out", "workspace"],
            [5, 7, 4, 8, 9, 10, 11, 12],
        ),
    ]
    for index, config in enumerate(options.values()):
        code += f"case {index}: {{\n"
        for stage, (op, epsilon, names, indices) in enumerate(stages):
            code += ("int status = " if stage == 0 else "return ") + "[&]() -> int {\n"
            for name, i in zip(names, indices):
                kind = "uint8_t*" if name == "workspace" else "void*"
                code += f"auto {name}=static_cast<{kind}>(p[{i}]);\n"
            is_target = op == attrs["op"]
            code += call_body(
                dict(op=op, joint=True, eps=epsilon),
                arch,
                config if is_target else None,
                eps="eps" if is_target else f"{epsilon}f",
            )
            code += "\n}();\n"
            if stage == 0:
                code += "if(status) return status;\n"
        code += "}\n"
    code += "default: return -1; } }\n"
    code += f"int main(int argc,char** argv) {{ return profile_main(argc,argv,{len(options)},specs,{{4,5,11}},candidate); }}\n"
    return code
