"""AOT code generation for residual/RMS/ReLU projection fusion."""

from pathlib import Path
import subprocess
from aitemplate.backend import registry
from aitemplate.backend.target import Target


def _signature(name):
    return (
        f"void {name}(void* x, void* w, void* skip, void* gamma, void* up, "
        "void* outer, void* out, uint8_t* workspace, int64_t rows, cudaStream_t stream)"
    )


@registry.reg("cuda.gemm_rms_relu_gemm.gen_function")
def gen_function(attrs):
    arch = int(Target.current()._arch)
    if arch not in (90, 100):
        raise NotImplementedError("Fused RMS projections require SM90 or SM100")
    from . import native_fused_gemm

    if native_fused_gemm.enabled():
        return native_fused_gemm.boundary(attrs, arch)
    from .cutedsl_rms_relu_gemm import export

    name, directory = attrs["name"], Path(attrs["workdir"])
    if arch == 90:
        dispatch = [
            (2592, 64, False, 64),
            (10368, 128, False, 128),
            (None, 128, True, 192),
        ]
    else:
        dispatch = [
            (2592, 128, False, 64),
            (10368, 128, False, 128),
            (None, 128, False, 192),
        ]
    variants = {}
    for _, tm, pp, un in dispatch:
        for producer, tile_m, tile_n in ((True, tm, 192), (False, 128, un)):
            key = (producer, tile_m, tile_n, pp)
            if key not in variants:
                variant = (
                    f"{name}_{'p' if producer else 'c'}_{tile_m}_{tile_n}_{int(pp)}"
                )
                variants[key] = variant
                export(
                    directory, variant, arch, producer, tile_m, tile_n, pp, attrs["eps"]
                )
    combined = directory / (name + "_cutedsl.o")
    subprocess.run(
        [
            "ld",
            "-r",
            "-o",
            str(combined),
            *[str(directory / (v + ".o")) for v in variants.values()],
        ],
        check=True,
    )
    attrs["cutedsl_obj_path"] = str(combined)
    code = "#include <cuda_runtime.h>\n#include <cstdint>\n#include <mutex>\n#include <stdexcept>\n"
    for v in variants.values():
        code += f'#include "{v}.h"\n'
    code += "namespace {\n"
    for (producer, _, _, _), v in variants.items():
        k, n = (576, 192) if producer else (192, 384)
        # R is a matrix reduction output in the producer and a vector load in the consumer.
        rinit = "{r,{(int32_t)m,1},{1}}" if producer else "{r,{(int32_t)m}}"
        code += f"""
static {v}_Kernel_Module_t module_{v};
void call_{v}(void* x,void* w,void* c,void* d,void* r,void* g,int64_t m,cudaStream_t stream) {{
 {v}_Tensor_A_t A{{x,{{(int32_t)m,{k}}},{{{k}}}}};
 {v}_Tensor_B_t B{{w,{{{n},{k}}},{{{k}}}}};
 {v}_Tensor_C_t C{{c,{{(int32_t)m,{n}}},{{{n}}}}};
 {v}_Tensor_D_t D{{d,{{(int32_t)m,{n}}},{{{n}}}}};
 {v}_Tensor_R_t R{rinit};
 {v}_Tensor_G_t G{{g,{{192}}}};
 int result=cute_dsl_{v}_wrapper(&module_{v},&A,&B,&C,&D,&R,&G,stream);
 if(result)throw std::runtime_error("Fused RMS projection launch failed");
}}
"""
    code += (
        "}\n"
        + _signature(name)
        + " {\n static std::once_flag once;\n std::call_once(once,[]{\n"
    )
    for v in variants.values():
        code += f"  {v}_Kernel_Module_Load(&module_{v});\n"
    code += (
        " });\n void* tmp=workspace;\n void* r=workspace+((rows*384+255)/256)*256;\n"
    )
    for i, (bound, tm, pp, un) in enumerate(dispatch):
        prefix = "else " if i else ""
        condition = f"if(rows<={bound}) " if bound is not None else ""
        p = variants[(True, tm, 192, pp)]
        c = variants[(False, 128, un, pp)]
        code += (
            f" {prefix}{condition}{{\n call_{p}(x,w,skip,tmp,r,gamma,rows,stream);\n"
        )
        code += f" call_{c}(tmp,up,outer,out,r,gamma,rows,stream);\n }}\n"
    from .cutedsl_loader import register_loader

    return register_loader(code + "}\n", _signature(name), name)


@registry.reg("cuda.gemm_rms_relu_gemm.func_decl")
def gen_decl(attrs):
    return _signature(attrs["name"]) + ";"


@registry.reg("cuda.gemm_rms_relu_gemm.func_call")
def gen_call(attrs, indent="  "):
    names = [t._attrs["name"] for t in attrs["inputs"] + attrs["outputs"]]
    rows = " * ".join(
        str(d.value()) if len(d._attrs["values"]) == 1 else d._attrs["name"]
        for d in attrs["inputs"][0].shape()[:-1]
    )
    names += ["global_workspace_", rows, "stream"]
    return indent + attrs["name"] + "(" + ", ".join(names) + ");"
