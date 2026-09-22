"""FP16 RMS reduction fused into RoPE/unpack or SwiGLU; no intermediate scale tensor."""
import jinja2
from aitemplate.backend import registry

_KERNEL = jinja2.Template(r'''
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
namespace {
__device__ float {{name}}_rms(const half* x, int64_t row, int lane) {
  float ss = 0.f;
  for (int chunk = lane; chunk < {{channels // 8}}; chunk += 32) {
    uint4 q = reinterpret_cast<const uint4*>(x + row * {{channels}})[chunk];
    const half2* h = reinterpret_cast<const half2*>(&q);
    #pragma unroll
    for (int i=0;i<4;++i) {
      float2 f=__half22float2(h[i]);
      ss += f.x*f.x+f.y*f.y;
    }
  }
  #pragma unroll
  for (int d=16;d;d>>=1) ss+=__shfl_xor_sync(0xffffffff,ss,d);
  return __half2float(__float2half_rn(rsqrtf(ss/{{channels}}.f+{{eps}}f)));
}

{% if kind == "swiglu" %}
__global__ void {{name}}_swiglu(const half* x, const half* gateup, half* out, int64_t rows) {
  int64_t row=int64_t(blockIdx.x)*(blockDim.x/32)+threadIdx.x/32;
  int lane=threadIdx.x%32;
  if(row>=rows) return;
  float r={{name}}_rms(x,row,lane);
  #pragma unroll
  for(int j=lane;j<{{ffn // 8}};j+=32) {
    uint4 g=reinterpret_cast<const uint4*>(gateup+row*{{2 * ffn}})[j];
    uint4 u=reinterpret_cast<const uint4*>(gateup+row*{{2 * ffn}}+{{ffn}})[j];
    half2* gh=reinterpret_cast<half2*>(&g);
    const half2* uh=reinterpret_cast<const half2*>(&u);
    #pragma unroll
    for(int i=0;i<4;++i) {
      float2 gf=__half22float2(gh[i]), uf=__half22float2(uh[i]);
      float a=gf.x*r,b=gf.y*r;
      gh[i]=__floats2half2_rn((a/(1.f+__expf(-a)))*(uf.x*r),(b/(1.f+__expf(-b)))*(uf.y*r));
    }
    reinterpret_cast<uint4*>(out+row*{{ffn}})[j]=g;
  }
}

{% else %}
__global__ void {{name}}_rope(const half* x,const half* qkv,const half* co,const half* si,
                          half* q,half* k,half* v,int64_t rows) {
  int64_t row=int64_t(blockIdx.x)*(blockDim.x/32)+threadIdx.x/32;
  int lane=threadIdx.x%32;
  if(row>=rows) return;
  float r={{name}}_rms(x,row,lane);
  // 24 vector chunks per token; two chunks per head, four rotary pairs each.
  for(int chunk=lane;chunk<{{dim // 8}};chunk+=32) {
    uint2 c=reinterpret_cast<const uint2*>(co+(row%{{seq}})*{{dim // 2}})[chunk];
    uint2 s=reinterpret_cast<const uint2*>(si+(row%{{seq}})*{{dim // 2}})[chunk];
    const half* ch=reinterpret_cast<const half*>(&c);
    const half* sh=reinterpret_cast<const half*>(&s);
    #pragma unroll
    for(int which=0;which<3;++which) {
      uint4 a=reinterpret_cast<const uint4*>(qkv+row*{{3 * dim}}+which*{{dim}})[chunk];
      half2* h=reinterpret_cast<half2*>(&a);
      #pragma unroll
      for(int j=0;j<4;++j) {
        float2 z=__half22float2(h[j]);z.x*=r;z.y*=r;
        float cf=__half2float(ch[j]),sf=__half2float(sh[j]);
        h[j]=__floats2half2_rn(which<2?z.x*cf-z.y*sf:z.x,which<2?z.x*sf+z.y*cf:z.y);
      }
      half* dst=which==0?q:(which==1?k:v);
      reinterpret_cast<uint4*>(dst+row*{{dim}})[chunk]=a;
    }
  }
}


{% endif %}
}
void {{name}}(const void* source, const void* projected,
{% if kind == "rope" %}const void* co, const void* si, void* q, void* k, void* v,
{% else %}void* output,
{% endif %}int64_t rows, cudaStream_t stream) {
{% if kind == "rope" %}
  {{name}}_rope<<<(rows+3)/4,128,0,stream>>>((const half*)source,(const half*)projected,
      (const half*)co,(const half*)si,(half*)q,(half*)k,(half*)v,rows);
{% else %}
  {{name}}_swiglu<<<(rows+3)/4,128,0,stream>>>((const half*)source,(const half*)projected,(half*)output,rows);
{% endif %}
}
''')


def _generate(attrs):
    rope = attrs["op"] == "rms_unpack_rope"
    projected = attrs["inputs"][0]
    source = attrs["inputs"][-1]
    return _KERNEL.render(
        name=attrs["name"], kind="rope" if rope else "swiglu",
        channels=source.shape()[-1].value(), eps=repr(attrs["eps"]),
        dim=attrs.get("dim", 0), seq=projected.shape()[1].value() if rope else 0,
        ffn=projected.shape()[-1].value() // 2 if not rope else 0,
    )


def _decl(attrs):
    args = ("const void*, const void*, const void*, const void*, void*, void*, void*, "
            if attrs["op"] == "rms_unpack_rope" else "const void*, const void*, void*, ")
    return "void " + attrs["name"] + "(" + args + "int64_t, cudaStream_t);\n"


def _call(attrs, indent="  "):
    inputs, outputs = attrs["inputs"], attrs["outputs"]
    ordered = [inputs[-1]] + inputs[:-1] + outputs
    args = [t._attrs["name"] for t in ordered]
    args += [" * ".join(d._attrs["name"] for d in inputs[0].shape()[:-1]) or "1", "stream"]
    return indent + attrs["name"] + "(" + ", ".join(args) + ");\n"


for _op in ("rms_swiglu", "rms_unpack_rope"):
    registry.reg("cuda." + _op + ".gen_function")(_generate)
    registry.reg("cuda." + _op + ".func_decl")(_decl)
    registry.reg("cuda." + _op + ".func_call")(_call)
