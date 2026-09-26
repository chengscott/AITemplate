"""AOT FP16 residual/RMS/SwiGLU fusion for supported SM80/SM90/SM100 shapes."""

from pathlib import Path
import subprocess
from aitemplate.backend import registry
from aitemplate.backend.target import Target


def _signature(name):
    return (
        f"void {name}(void* x, void* w, void* skip, void* proj, void* updated, "
        "void* out, uint8_t* workspace, int64_t rows, cudaStream_t stream)"
    )


@registry.reg("cuda.gemm_rms_swiglu.gen_function")
def gen_function(attrs):
    arch = int(Target.current()._arch)
    if arch not in (80, 90, 100):
        raise NotImplementedError("Fused RMS GEMM supports SM80, SM90 and SM100")
    from . import native_fused_gemm

    if native_fused_gemm.enabled():
        return native_fused_gemm.swiglu(attrs, arch)
    if arch == 80:
        raise NotImplementedError("SM80 fused RMS GEMM requires the cutlass backend")

    from .cutedsl_rms_swiglu import export

    name, directory = attrs["name"], Path(attrs["workdir"])
    # (row upper bound, M tile, N tile, ping-pong schedule).
    if arch == 90:
        ffn_dispatch = [
            (324, 64, 64, False),
            (648, 128, 64, True),
            (1296, 128, 128, True),
            (None, 128, 192, True),
        ]
        residual_dispatch = [(5184, 64, 192, False), (None, 128, 192, False)]
    else:
        ffn_dispatch = [
            (648, 128, 64, False),
            (10368, 128, 128, False),
            (None, 128, 192, True),
        ]
        residual_dispatch = [(None, 128, 192, True)]
    variants, objects = [], []
    dispatches = [("ffn", ffn_dispatch)]
    if attrs["joint"]:
        dispatches.append(("res", residual_dispatch))
    for kind, dispatch in dispatches:
        for _, tile_m, tile_n, pingpong in dispatch:
            variant = f"{name}_{kind}_{tile_m}_{tile_n}_{int(pingpong)}"
            residual = kind == "res"
            objects.append(
                export(
                    directory,
                    variant,
                    arch,
                    tile_n,
                    residual,
                    tile_m=tile_m,
                    pingpong=pingpong,
                    direct_rms=residual,
                    eps=attrs["eps"],
                )
            )
            variants.append((variant, residual, tile_n))
    combined = directory / (name + "_cutedsl.o")
    subprocess.run(["ld", "-r", "-o", str(combined), *objects], check=True)
    attrs["cutedsl_obj_path"] = str(combined)
    code = "#include <cuda_runtime.h>\n#include <cuda_fp16.h>\n#include <cstdint>\n#include <mutex>\n#include <stdexcept>\n"
    for v, _, _ in variants:
        code += f'#include "{v}.h"\n'
    code += "namespace {\n"
    for v, residual, tile_n in variants:
        code += f"static {v}_Kernel_Module_t module_{v};\n"
        if residual:
            code += f"""
void call_{v}(void* x,void* w,void* skip,void* out,void* partial,int64_t m,cudaStream_t stream) {{
 {v}_Tensor_A_t A{{x,{{(int32_t)m,192}},{{192}}}};
 {v}_Tensor_B_t B{{w,{{192,192}},{{192}}}};
 {v}_Tensor_C_t C{{skip,{{(int32_t)m,192}},{{192}}}};
 {v}_Tensor_D_t D{{out,{{(int32_t)m,192}},{{192}}}};
 {v}_Tensor_P_t P{{partial,{{(int32_t)m,{192//tile_n}}},{{{192//tile_n}}}}};
 int result=cute_dsl_{v}_wrapper(&module_{v},&A,&B,&C,&D,&P,stream);
 if(result)throw std::runtime_error("Fused residual GEMM launch failed");
}}
"""
        else:
            code += f"""
void call_{v}(void* x,void* w,void* r,void* out,int64_t m,cudaStream_t stream) {{
 {v}_Tensor_A_t A{{x,{{(int32_t)m,192}},{{192}}}};
 {v}_Tensor_B_t B{{w,{{1152,192}},{{192}}}};
 {v}_Tensor_R_t R{{r,{{(int32_t)m}}}};
 {v}_Tensor_O_t O{{out,{{(int32_t)m,576}},{{576}}}};
 int result=cute_dsl_{v}_wrapper(&module_{v},&A,&B,&R,&O,stream);
 if(result)throw std::runtime_error("Fused SwiGLU GEMM launch failed");
}}
"""
    eps = attrs["eps"]
    code += f"""
__global__ void {name}_rms(const half* x,float* out,int64_t m) {{
 int64_t row=int64_t(blockIdx.x)*4+threadIdx.x/32;
 int lane=threadIdx.x%32;if(row>=m)return;float ss=0;
 if(lane<24) {{
  uint4 q=reinterpret_cast<const uint4*>(x+row*192)[lane];
  const half2* h=reinterpret_cast<const half2*>(&q);
  #pragma unroll
  for(int i=0;i<4;++i){{float2 v=__half22float2(h[i]);ss+=v.x*v.x+v.y*v.y;}}
 }}
 #pragma unroll
 for(int d=16;d;d>>=1)ss+=__shfl_xor_sync(0xffffffff,ss,d);
 if(lane==0)out[row]=__half2float(__float2half_rn(rsqrtf(ss/192.f+{eps}f)));
}}
}} // namespace
{_signature(name)} {{
 static std::once_flag once;
 std::call_once(once,[]{{
"""
    for v, _, _ in variants:
        code += f"  {v}_Kernel_Module_Load(&module_{v});\n"
    code += " });\n"

    def emit_dispatch(kind, dispatch, args):
        result = ""
        for i, (bound, tm, tn, pp) in enumerate(dispatch):
            prefix = "else " if i else ""
            condition = f"if(rows<={bound})" if bound is not None else ""
            result += (
                f" {prefix}{condition}call_{name}_{kind}_{tm}_{tn}_{int(pp)}({args});\n"
            )
        return result

    if attrs["joint"]:
        code += emit_dispatch(
            "res", residual_dispatch, "x,proj,skip,updated,workspace,rows,stream"
        )
        code += " x=updated;\n"
    else:
        code += f"{name}_rms<<<(rows+3)/4,128,0,stream>>>((half*)x,(float*)workspace,rows);\n"
        code += ' if(cudaGetLastError()!=cudaSuccess)throw std::runtime_error("Fused RMS reduction launch failed");\n'
    code += emit_dispatch("ffn", ffn_dispatch, "x,w,workspace,out,rows,stream")
    code += "}\n"
    from .cutedsl_loader import register_loader

    return register_loader(code, _signature(name), name)


@registry.reg("cuda.gemm_rms_swiglu.func_decl")
def gen_decl(attrs):
    return _signature(attrs["name"]) + ";"


@registry.reg("cuda.gemm_rms_swiglu.func_call")
def gen_call(attrs, indent="  "):
    x, w = attrs["inputs"][:2]
    names = [x._attrs["name"], w._attrs["name"]]
    if attrs["joint"]:
        names += [t._attrs["name"] for t in attrs["inputs"][2:]]
        names += [attrs["outputs"][0]._attrs["name"]]
    else:
        names += ["nullptr"] * 3
    names += [attrs["outputs"][-1]._attrs["name"], "global_workspace_"]
    dims = x.shape()[:-1]
    rows = " * ".join(
        str(d.value()) if len(d._attrs["values"]) == 1 else d._attrs["name"]
        for d in dims
    )
    names += [rows, "stream"]
    return indent + attrs["name"] + "(" + ", ".join(names) + ");"
