"""Register AOT GEMM modules for loading before CUDA graph capture."""


def register_loader(code, signature, name):
    marker = signature + " {"
    start = code.index(marker)
    init_start = start + len(marker)
    init_end = code.index(" });", init_start) + len(" });")
    initializer = code[init_start:init_end]
    loader = f"load_{name}"
    registration = f"""
namespace {{
void {loader}() {{{initializer}
}}
struct Register_{name} {{
 Register_{name}() {{ ait::_cutedsl_loaders().push_back(&{loader}); }}
}} register_{name};
}}
{marker}
 {loader}();
"""
    return (
        "#include <vector>\nnamespace ait { extern std::vector<void (*)()>& _cutedsl_loaders(); }\n"
        + code[:start]
        + registration
        + code[init_end:]
    )
