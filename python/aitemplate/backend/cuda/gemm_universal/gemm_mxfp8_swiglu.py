"""AOT MXFP8 FFN with producer-side activation and block quantization."""

from pathlib import Path
import subprocess
from aitemplate.backend import registry
from aitemplate.backend.target import Target

# Eight lanes share a row; six lanes quantize its 32-element blocks.
_QUANTIZE_SRC = r"""
__global__ void prepare_quant(const __half* __restrict__ a,
                                    __half* __restrict__ rrms_out, float eps,
                                    unsigned char* __restrict__ aq,
                                    unsigned char* __restrict__ sfa, long long rows) {
  constexpr int nkb = 6, ntx = 2;

  const int warps_per_cta = blockDim.x / 8;
  const long long row = (long long)blockIdx.x * warps_per_cta + (threadIdx.x / 8);
  if (row >= rows) return;
  const int lane = threadIdx.x % 8;
  const long long K = (long long)nkb * 32;
  const uint4* ar = reinterpret_cast<const uint4*>(a + row * K);
  uint2* aqr = reinterpret_cast<uint2*>(aq + row * K);
  const long long iy = row & 127;

  uint4 xbuf[1][4]; float ss = 0.f; int bi = 0;
#pragma unroll
  for (int kb = lane; kb < nkb; kb += 8, bi++) {
#pragma unroll
    for (int j = 0; j < 4; j++) { uint4 q = ar[kb * 4 + j]; xbuf[bi][j] = q; const __half2* h = reinterpret_cast<const __half2*>(&q);
#pragma unroll
      for (int i = 0; i < 4; i++) { float2 f = __half22float2(h[i]); ss += f.x * f.x + f.y * f.y; } }
  }
#pragma unroll
  for (int o = 4; o > 0; o >>= 1) ss += __shfl_xor_sync(__activemask(), ss, o);
  const float rrms = rsqrtf(ss / (float)K + eps);
  if (lane == 0) rrms_out[row] = __float2half(rrms);
  bi = 0;
#pragma unroll
  for (int kb = lane; kb < nkb; kb += 8, bi++) {
    float amax = 0.f;
#pragma unroll
    for (int j = 0; j < 4; j++) { const __half2* h = reinterpret_cast<const __half2*>(&xbuf[bi][j]);
#pragma unroll
      for (int i = 0; i < 4; i++) { float2 f = __half22float2(h[i]); amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y))); } }
    int b = amax > 0.f ? ((int)ceilf(log2f(amax * (1.0f / 448.0f))) + 127) : 0; b = b < 0 ? 0 : (b > 254 ? 254 : b);
    float inv = amax > 0.f ? exp2f((float)(127 - b)) : 0.f;
    sfa[(row >> 7) * (long long)ntx * 512 + (kb >> 2) * 512 + (iy % 32) * 16 + (iy >> 5) * 4 + (kb & 3)] = (unsigned char)b;
#pragma unroll
    for (int j = 0; j < 4; j++) { const __half2* h = reinterpret_cast<const __half2*>(&xbuf[bi][j]);
      uint2 out; unsigned short* os = reinterpret_cast<unsigned short*>(&out);
#pragma unroll
      for (int i = 0; i < 4; i++) { float2 f = __half22float2(h[i]); f.x *= inv; f.y *= inv; os[i] = __nv_fp8x2_e4m3(f).__x; }
      aqr[kb * 4 + j] = out; }
  }

}

"""


def _signature(name):
    return f"void {name}(void* x,void* w,void* sf,void* down,void* down_sf,void* out,uint8_t* workspace,int64_t rows,cudaStream_t stream)"


@registry.reg("cuda.gemm_mxfp8_swiglu.gen_function")
def gen_function(attrs):
    if Target.current()._arch != "100":
        raise NotImplementedError("MXFP8 SwiGLU producer fusion requires SM100")
    from . import native_fused_gemm

    if native_fused_gemm.enabled():
        return native_fused_gemm.mxfp8(attrs)
    from .cutedsl_mxfp8_swiglu import export

    name, directory = attrs["name"], Path(attrs["workdir"])
    variants = {}
    for producer, n in ((True, 64), (True, 128), (True, 256), (False, 192)):
        v = f"{name}_{'p'if producer else'c'}_{n}"
        variants[(producer, n)] = v
        export(directory, v, producer, n)
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
    code = "#include <cuda_runtime.h>\n#include <cuda_fp16.h>\n#include <cuda_fp8.h>\n#include <cstdint>\n#include <cmath>\n#include <mutex>\n#include <stdexcept>\n"
    for v in variants.values():
        code += f'#include "{v}.h"\n'
    code += "namespace {\n"
    code += _QUANTIZE_SRC
    for (producer, n), v in variants.items():
        k = 192 if producer else 576
        width = 1152 if producer else 192
        rk = (k + 127) // 128
        rn = (width + 127) // 128
        code += f"static {v}_Kernel_Module_t module_{v};\n"
        code += f"void call_{v}(void* a,void* b,void* c,void* out,void* r,void* sa,void* sb,void* sh,int64_t m,cudaStream_t stream) {{\n"
        code += f" {v}_Tensor_A_t A{{a,{{(int32_t)m,{k}}},{{{k}}}}};\n {v}_Tensor_B_t B{{b,{{{width},{k}}},{{{k}}}}};\n"
        code += f" int32_t rm=(m+127)/128;\n {v}_Tensor_SA_t SA{{sa,{{rm,{rk}}},{{int64_t(rm)*{rk}*512,{rk}*512,512}}}};\n"
        code += (
            f" {v}_Tensor_SB_t SB{{sb,{{{rn},{rk}}},{{{rn*rk*512},{rk*512},512}}}};\n"
        )
        if producer:
            code += f" {v}_Tensor_R_t R{{r,{{(int32_t)m}}}};\n {v}_Tensor_H_t H{{out,{{(int32_t)m,576}},{{576}}}};\n"
            code += (
                f" {v}_Tensor_SH_t SH{{sh,{{rm,5}},{{int64_t(rm)*2560,2560,512}}}};\n"
            )
            args = "&A,&B,&R,&H,&SA,&SB,&SH"
        else:
            code += f" {v}_Tensor_C_t C{{c,{{(int32_t)m,192}},{{192}}}};\n {v}_Tensor_D_t D{{out,{{(int32_t)m,192}},{{192}}}};\n"
            args = "&A,&B,&C,&D,&SA,&SB"
        code += f' int result=cute_dsl_{v}_wrapper(&module_{v},{args},stream);\n if(result)throw std::runtime_error("MXFP8 FFN launch failed");\n}}\n'
    code += (
        "}\n"
        + _signature(name)
        + " {\n static std::once_flag once;\n std::call_once(once,[]{\n"
    )
    for v in variants.values():
        code += f" {v}_Kernel_Module_Load(&module_{v});\n"
    code += " });\n auto aligned=[](int64_t n){return (n+255)/256*256;};\n int64_t rm=(rows+127)/128;\n"
    code += " void* aq=workspace;\n void* sa=(uint8_t*)aq+aligned(rows*192);\n void* r=(uint8_t*)sa+rm*1024;\n void* h=(uint8_t*)r+aligned(rows*2);\n void* sh=(uint8_t*)h+aligned(rows*576);\n"
    code += f" prepare_quant<<<(rows+31)/32,256,0,stream>>>((half*)x,(half*)r,{attrs['eps']}f,(unsigned char*)aq,(unsigned char*)sa,rows);\n"
    code += ' if(cudaGetLastError()!=cudaSuccess)throw std::runtime_error("MXFP8 input quantization failed");\n'
    for i, (bound, n) in enumerate(((648, 64), (10368, 128), (None, 256))):
        v = variants[(True, n)]
        prefix = "else " if i else ""
        condition = f"if(rows<={bound}) " if bound else ""
        code += (
            f" {prefix}{condition}call_{v}(aq,w,nullptr,h,r,sa,sf,sh,rows,stream);\n"
        )
    code += f" call_{variants[(False,192)]}(h,down,x,out,nullptr,sh,down_sf,nullptr,rows,stream);\n}}\n"
    from .cutedsl_loader import register_loader

    return register_loader(code, _signature(name), name)


@registry.reg("cuda.gemm_mxfp8_swiglu.func_decl")
def gen_decl(attrs):
    return _signature(attrs["name"]) + ";"


@registry.reg("cuda.gemm_mxfp8_swiglu.func_call")
def gen_call(attrs, indent="  "):
    args = [t._attrs["name"] for t in attrs["inputs"] + attrs["outputs"]]
    rows = " * ".join(
        str(d.value()) if len(d._attrs["values"]) == 1 else d._attrs["name"]
        for d in attrs["inputs"][0].shape()[:-1]
    )
    return (
        indent
        + attrs["name"]
        + "("
        + ", ".join(args + ["global_workspace_", rows, "stream"])
        + ");"
    )


@registry.reg("cuda.pack_mxfp8_swiglu.gen_function")
def pack_gen(attrs):
    name = attrs["name"]
    return f"""
#include <cuda_runtime.h>
#include <cstdint>
namespace {{
__global__ void {name}_kernel(const uint8_t* w,const uint8_t* sf,uint8_t* pw,uint8_t* ps) {{
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<1152*192) {{int row=i/192, col=i%192;int old=row/2+(row%2)*576;pw[i]=w[old*192+col];}}
 if(i<1152*8) {{
  int row=i/8,kb=i%8,old=row/2+(row%2)*576;
  int dst=(row/128)*1024+(kb/4)*512+(row%32)*16+((row%128)/32)*4+kb%4;
  int src=(old/128)*1024+(kb/4)*512+(old%32)*16+((old%128)/32)*4+kb%4;
  ps[dst]=sf[src];
 }}
}}
}}
void {name}(const void* w,const void* sf,void* pw,void* ps,cudaStream_t stream) {{
 {name}_kernel<<<(1152*192+255)/256,256,0,stream>>>((const uint8_t*)w,(const uint8_t*)sf,(uint8_t*)pw,(uint8_t*)ps);
}}
"""


@registry.reg("cuda.pack_mxfp8_swiglu.func_decl")
def pack_decl(attrs):
    return f"void {attrs['name']}(const void*,const void*,void*,void*,cudaStream_t);"


@registry.reg("cuda.pack_mxfp8_swiglu.func_call")
def pack_call(attrs, indent="  "):
    args = [t._attrs["name"] for t in attrs["inputs"] + attrs["outputs"]]
    return indent + attrs["name"] + "(" + ", ".join(args + ["stream"]) + ");"
