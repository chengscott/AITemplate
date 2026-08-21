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
"""
Util functions to handle alignment.
"""

from typing import List

from aitemplate.compiler.dtype import normalize_dtype


# FIXME: These alignment constraints are for cutlass/ck. We should consider
# to refine this part for other backends.
def get_alignments(dtype: str) -> List[int]:
    """
    Return all of the valid alignment values for the dtype.
    """
    dtype = normalize_dtype(dtype)
    if dtype in ("float16", "bfloat16"):
        return [8, 4, 2, 1]
    elif dtype in ("float", "float32"):
        return [4, 2, 1]
    elif dtype == "float8_e4m3":
        # 1 byte/elem: 16-elem alignment == 16 bytes (the Hopper TMA requirement
        # for fp8 conv/gemm operands). Used to size A/B alignment for the SM90 3.x
        # fp8 conv path (see backend/cuda/conv2d/common.py).
        return [16, 8, 4, 2, 1]
    else:
        raise NotImplementedError(f"unsupported {dtype=} for alignments")


def find_max_alignment(number: int, dtype: str) -> int:
    """
    Return the first alignment value that meets the alignment requirement
    for accessing the `number` of elements. This is dtype dependent.
    """
    alignments = get_alignments(dtype)
    for alignment in alignments:
        if number % alignment == 0:
            return alignment
    return 1


def find_max_alignment_from(numbers: List[int], dtype: str) -> int:
    """
    Return the max alignment value that is valid for all the numbers.
    """
    alignments = get_alignments(dtype)
    for alignment in alignments:
        if all(number % alignment == 0 for number in numbers):
            return alignment
    return 1


def valid_alignment(align: int, dtype: str) -> bool:
    """
    Return True if the given align value is legitimate for the dtype.
    """
    dtype = normalize_dtype(dtype)
    # 2-elem-alignment is required by fp16, because async.copy needs at least 32
    # bits. For fp32 dtype values, 1-elem-alignment is valid.
    if dtype in ("float16", "bfloat16"):
        return align % 2 == 0
    elif dtype in ("float", "float32"):
        return True
    elif dtype == "float8_e4m3":
        # fp8 Hopper TMA wants 16-byte (=16-elem) alignment; a shorter last dim is
        # padded up to the next 16 (apply_padding). All trunk conv channel counts
        # (128/256) are already 16-aligned, so this pads nothing in practice.
        return align % 16 == 0
    else:
        raise NotImplementedError(f"unsupported {dtype=} for valid_alignment")
