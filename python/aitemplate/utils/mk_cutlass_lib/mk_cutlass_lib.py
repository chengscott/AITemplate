#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import os
import pathlib
import re
import shutil
import tempfile

from aitemplate.utils.mk_cutlass_lib import (
    extra_conv_emit,
    extra_cutlass_generator,
    extra_enum,
    extra_gemm_emit,
)


def mk_cutlass_lib(template_path, dst_prefix=None):
    if dst_prefix is None:
        dst_prefix = tempfile.mkdtemp()
    lib_dst = os.path.join(dst_prefix, "cutlass_lib")
    if pathlib.Path(lib_dst).is_dir():
        shutil.rmtree(lib_dst)

    os.makedirs(lib_dst)
    with open(os.path.join(lib_dst, "__init__.py"), "w") as fo:
        fo.write("from . import library\n")
        fo.write("from . import generator\n")
        fo.write("from . import manifest\n")
        fo.write("from . import conv3d_operation\n")
        fo.write("from . import gemm_operation\n")
        fo.write("from . import conv2d_operation\n")
        fo.write("from . import extra_operation\n")

    def process_code(src_path, dst_path, code_set):
        # Rewrite intra-package imports to explicit relative imports so that the
        # generated cutlass_lib package is self-contained and uses these patched
        # copies rather than any installed `cutlass_library` package.
        #
        # cutlass >= 3.2 ships the generator under python/cutlass_library and
        # guards its imports as:
        #     try:
        #       from cutlass_library.library import *
        #     except ImportError:
        #       from library import *
        # Both the `cutlass_library.<mod>` and bare `<mod>` forms (with `import *`
        # or explicit names) are normalized to `from .<mod> import ...`, and
        # module-style `import cutlass_library.<mod>` / `import <mod>` (cutlass
        # >= 4.x, e.g. the SM100 helpers) become `from . import <mod>`.
        from_pattern = re.compile(
            r"^(\s*)from\s+(?:cutlass_library\.)?([a-z_0-9]+)\s+import\s+(.*)$"
        )
        import_pattern = re.compile(
            r"^(\s*)import\s+(?:cutlass_library\.)?([a-z_0-9]+)\s*$"
        )
        with open(src_path) as fi:
            lines = fi.readlines()
        output = []

        for line in lines:
            match = from_pattern.match(line)
            if match is not None:
                indent, name, rest = match.groups()
                if name + ".py" in code_set:
                    line = "{indent}from .{name} import {rest}\n".format(
                        indent=indent, name=name, rest=rest
                    )
            else:
                match = import_pattern.match(line)
                if match is not None:
                    indent, name = match.groups()
                    if name + ".py" in code_set:
                        line = "{indent}from . import {name}\n".format(
                            indent=indent, name=name
                        )
            output.append(line)
        if "library.py" in dst_path:
            lines = extra_enum.emit_library()
            output.append(lines)
        if "conv2d_operation.py" in dst_path:
            lines = extra_conv_emit.emit_library()
            output.append(lines)
        if "gemm_operation.py" in dst_path:
            lines = extra_gemm_emit.emit_library()
            output.append(lines)
        with open(dst_path, "w") as fo:
            fo.writelines(output)

    src_prefix = os.path.join(template_path, "python/cutlass_library")
    srcs = os.listdir(src_prefix)
    if "__init__.py" in srcs:
        srcs.remove("__init__.py")
    for file in srcs:
        src_path = os.path.join(src_prefix, file)
        if not os.path.isfile(src_path):
            continue
        dst_path = os.path.join(lib_dst, file)
        process_code(src_path, dst_path, srcs)

    # extra configs
    dst_path = os.path.join(lib_dst, "extra_operation.py")
    with open(dst_path, "w") as fo:
        code = extra_cutlass_generator.emit_library()
        fo.write(code)
    return dst_prefix


def main() -> None:
    cutlass_path = os.getenv("SRCDIR")
    output_path = os.getenv("OUT")

    assert output_path is not None
    assert cutlass_path is not None

    mk_cutlass_lib(cutlass_path + "/cutlass", os.path.dirname(output_path))


if __name__ == "__main__":
    main()  # pragma: no cover
