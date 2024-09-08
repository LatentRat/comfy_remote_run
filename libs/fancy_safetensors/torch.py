#!/usr/bin/env python3
#
# Copyright (c) 2022 LatentRat
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#

from __future__ import annotations

import json
import os
from typing import Iterable, Callable

import safetensors
import safetensors.torch
from safetensors import safe_open

from .fancy_safetensors import encode, decode, is_encoded
import torch


def load(
        path_or_data: str | os.PathLike | bytes, device = "cpu",
        *,
        keep_keys: str | Iterable[str] = None,
        keep_key_prefixes: str | Iterable[str] = None,
        keep_key_fn: Callable[[str], bool] = None,
        float_dtype: torch.dtype | None = None,
        return_metadata = False,
):
    state_dict, metadata = load_filtered(
        path_or_data, device,
        keep_keys = keep_keys,
        keep_key_prefixes = keep_key_prefixes,
        keep_key_fn = keep_key_fn,
        float_dtype = float_dtype,
        return_metadata = True,
    )
    if not metadata:
        raise ValueError("metadata not found")

    return decode(state_dict, metadata, return_metadata)


def load_maybe_encoded(
        path_or_data: str | os.PathLike | bytes, device = "cpu",
        *,
        keep_keys: str | Iterable[str] = None,
        keep_key_prefixes: str | Iterable[str] = None,
        keep_key_fn: Callable[[str], bool] = None,
        float_dtype: torch.dtype | None = None,
        return_metadata = False,
):
    state_dict, metadata = load_filtered(
        path_or_data, device,
        keep_keys = keep_keys,
        keep_key_prefixes = keep_key_prefixes,
        keep_key_fn = keep_key_fn,
        float_dtype = float_dtype,
        return_metadata = True,
    )
    if is_encoded(metadata):
        return decode(state_dict, metadata, return_metadata)

    if return_metadata:
        return state_dict, metadata
    return state_dict


def load_filtered(
        path_or_data: str | os.PathLike | bytes, device = "cpu",
        *,
        keep_keys: str | Iterable[str] = None,
        keep_key_prefixes: str | Iterable[str] = None,
        keep_key_fn: Callable[[str], bool] = None,
        float_dtype: torch.dtype | None = None,
        return_metadata = False,
):
    if keep_keys is not None:
        if isinstance(keep_keys, str):
            keep_keys = { keep_keys }
        keep_keys = set(keep_keys) or None

    if keep_key_prefixes is not None:
        if isinstance(keep_key_prefixes, str):
            keep_key_prefixes = (keep_key_prefixes,)
        keep_key_prefixes = tuple(keep_key_prefixes) or None

    state_dict = { }

    if isinstance(path_or_data, bytes):
        class _FileEmu():
            def __init__(self, state_dict):
                self.state_dict = state_dict

            def metadata(self):
                import struct
                length_of_header = struct.unpack('<Q', path_or_data[:8])[0]
                headers = path_or_data[8:8 + length_of_header].decode('utf-8')
                headers = json.loads(headers)
                return (headers or { }).get("__metadata__")

            def keys(self):
                return self.state_dict.keys()

            def get_tensor(self, key):
                return self.state_dict[key]

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        file = safetensors.torch.load(path_or_data)
        file = _FileEmu(file)
    else:
        file = safe_open(str(path_or_data), framework = "pt", device = str(device))

    with file:
        metadata = file.metadata() if return_metadata else None
        for key in file.keys():
            if keep_keys is not None and key not in keep_keys:
                continue
            if keep_key_prefixes is not None and not key.startswith(keep_key_prefixes):
                continue
            if keep_key_fn is not None and not keep_key_fn(key):
                continue

            tensor = file.get_tensor(key)
            if float_dtype is not None and tensor.is_floating_point():
                tensor = tensor.to(float_dtype)
            state_dict[key] = tensor

    if return_metadata:
        return state_dict, metadata
    return state_dict


def save_file(obj, file_path: str | os.PathLike, transform_tensor_fn = None, keep_dict_keys: bool = False):
    state_dict, metadata = encode(obj, transform_tensor_fn = transform_tensor_fn, keep_dict_keys = keep_dict_keys)
    safetensors.torch.save_file(state_dict, str(file_path), metadata)


def save_bytes(obj, transform_tensor_fn = None, keep_dict_keys: bool = False):
    state_dict, metadata = encode(obj, transform_tensor_fn = transform_tensor_fn, keep_dict_keys = keep_dict_keys)
    return safetensors.torch.save(state_dict, metadata)
