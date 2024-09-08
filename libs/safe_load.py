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

import dataclasses
import argparse
import collections
import zipfile
from zipfile import ZipFile
import functools

from typing import Union, Optional, Any, TypeVar
from collections.abc import Iterable, Collection, Sequence, Callable, Generator, AsyncGenerator

import torch
import pickletools

import logging as _logging

logger = _logging.getLogger(__name__)


class IGNORED_REDUCE():
    def __init__(self, ignored_name):
        self.ignored_name = ignored_name

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"IGNORED_REDUCE({self.ignored_name!r})"


def _skip_kv(allow_int_keys: bool, key, val, label: str):
    if isinstance(key, str):
        if key.startswith("__"):
            logger.info("%s ignoring __ keyval %r: %r", label, key, val)
            return True
        return False

    if allow_int_keys and isinstance(key, int):
        return False

    raise ValueError("unsupported key", label, key)


def pickle_bytes_safe_load_dict(
        pickle_bytes: bytes, persistent_id_load_fn,
        reduce_fns_custom = None,
        reduce_fns_ignore_unknown = False,
        extended: bool = True,
):
    reduce_fns = {
        **{ 'collections OrderedDict': collections.OrderedDict },
        **(reduce_fns_custom or { }),
    }
    stack = []
    memo = { }

    markobject = pickletools.markobject

    def stack_pop_until(end):
        items = []
        while True:
            item = stack.pop()
            if item is end:
                break
            items.append(item)
        return list(reversed(items))

    for opcode, arg, pos in pickletools.genops(pickle_bytes):
        # print((opcode.name, arg, pos))

        if opcode.name == "PROTO":
            # print("ignoring proto", opcode.name)
            continue

        if opcode.name == "STOP":
            break

        if opcode.name == "EMPTY_DICT":
            stack.append({ })
            continue

        if opcode.name in { "BINPUT", "LONG_BINPUT" }:
            # print("MEMO set", (opcode.arg, stack[-1]))
            memo[arg] = stack[-1]
            continue

        elif opcode.name in { "GET", "BINGET", "LONG_BINGET" }:
            stack.append(memo[arg])
            continue

        if opcode.name == "REDUCE":
            arg_tup = stack.pop()
            func_name = stack.pop()
            func = reduce_fns.get(func_name)
            if func is None:
                if reduce_fns_ignore_unknown:
                    logger.info("ignoring unkonwn reduce function %r with args %r", func_name, arg_tup)
                    stack.append(IGNORED_REDUCE(str(func_name)))
                    continue
                raise ValueError("unsupported reduce function", repr(func_name), arg_tup)

            # print("REDUCE", (func, arg_tup))
            item = func(*arg_tup)
            stack.append(item)
            continue

        if opcode.name == "EMPTY_LIST":
            stack.append([])
            continue

        if opcode.name == "EMPTY_TUPLE":
            stack.append(tuple())
            continue

        if opcode.name == "MARK":
            stack.append(markobject)
            continue

        if opcode.name == "NONE":
            stack.append(None)
            continue

        if opcode.name == "NEWTRUE":
            stack.append(True)
            continue

        if opcode.name == "NEWFALSE":
            stack.append(False)
            continue

        if extended and opcode.name == "BUILD":
            build_arg = stack.pop()
            last = stack[-1]
            if isinstance(last, dict) and isinstance(build_arg, dict):
                build_arg = { key: val for (key, val) in build_arg.items() if not _skip_kv(extended, key, val, "BUILD") }
                last.update(build_arg)
            else:
                logger.info("ignoring BUILD of object %r with args %r", last, build_arg)
            continue

        if opcode.name == "TUPLE":
            tup = tuple(stack_pop_until(markobject))
            stack.append(tup)
            continue

        if opcode.name == "APPENDS":
            values = stack_pop_until(markobject)
            target = stack[-1]
            if not isinstance(target, list):
                raise ValueError("expected list", type(target), target)
            target.extend(values)
            continue

        if opcode.name == "SETITEM":
            val = stack.pop()
            key = stack.pop()

            target = stack[-1]
            if not isinstance(target, dict):
                raise ValueError("expected settitems dict", type(target), target)

            if not _skip_kv(extended, key, val, "SETITEM"):
                target[key] = val
            continue

        if opcode.name == "SETITEMS":
            items = stack_pop_until(markobject)
            if len(items) % 2 != 0:
                raise ValueError("uneven SETITEMS key number", len(items), items)

            items = [(items[i], items[i + 1]) for i in range(0, len(items), 2)]
            set_map = dict(items)
            set_map = { key: val for (key, val) in set_map.items() if not _skip_kv(extended, key, val, "SETITEMS") }

            target = stack[-1]
            if not isinstance(target, dict):
                raise ValueError("expected settitems dict", type(target), target)

            target.update(set_map)
            continue

        if opcode.name == "TUPLE1":
            stack[-1] = tuple(stack[-1:])
            continue

        if opcode.name == "TUPLE2":
            stack[-2:] = [tuple(stack[-2:])]
            continue

        if opcode.name == "TUPLE3":
            stack[-3:] = [tuple(stack[-3:])]
            continue

        if opcode.name == "BINPERSID":
            persistent_id = stack.pop()
            stack.append(persistent_id_load_fn(persistent_id))
            continue

        if opcode.name == "APPEND":
            item = stack.pop()
            the_list = stack[-1]
            the_list.append(item)

            continue

        if opcode.name in {
            "BINUNICODE",
            "BININT1", "BININT2", "BININT", "LONG", "LONG1", "LONG4",
            "BINFLOAT",
            "GLOBAL",
        }:
            # print(f"OPT {opcode.name!r} pushing {arg !r}")
            stack.append(arg)
            continue

        raise ValueError("unsupported opcode", opcode.name, opcode)

    if len(stack) != 1:
        raise ValueError("invalid stack left", len(stack), stack)

    last = stack[0]
    if not isinstance(last, dict):
        raise ValueError("invalid last stack item not dict", type(last), last)

    return last


DTYPE_MAP = {
    "torch FloatStorage":  (torch.float32, 4),
    "torch HalfStorage":   (torch.float16, 2),
    "torch IntStorage":    (torch.int32, 4),
    "torch LongStorage":   (torch.int64, 8),
    "torch DoubleStorage": (torch.double, 8),
}


def _build_tensor(zipfile, archive_name, storage, storage_offset, size, stride, requires_grad, backward_hooks):
    if backward_hooks:
        raise ValueError("unsupported _rebuild_tensor_v2 arg", (storage_offset, stride, backward_hooks))

    (storage, dtype_str, index, location, element_count) = storage
    if storage != "storage":
        raise ValueError("expected storage", storage)

    dtype, dtype_size = DTYPE_MAP[dtype_str]
    data_path = f"{archive_name}/data/{index}"
    data = zipfile.read(data_path)

    expected_size = element_count * dtype_size
    if len(data) != expected_size:
        raise ValueError("read unexpected amount of bytes",
                         len(data), expected_size, data_path, element_count, dtype_size)

    tensor = torch.frombuffer(data, dtype = dtype, requires_grad = requires_grad)
    return tensor.set_(tensor, storage_offset = storage_offset, size = torch.Size(size), stride = stride)


def get_archive_name(zipfile: zipfile.ZipFile, required: bool, data_only: bool = True):
    names = set(zipfile.namelist())
    for file in zipfile.filelist:
        if "/" in file.filename:
            prefix = file.filename[:file.filename.index("/")]
            if not data_only:
                return prefix

            if f"{prefix}/data.pkl" in names:
                print(f"found {prefix=}")
                return prefix

    if required:
        raise ValueError("archive prefix not found")


def torch_safe_load_dict(model_path_or_zipfile: Union[str, zipfile.ZipFile], extended: bool = False):
    if isinstance(model_path_or_zipfile, str):
        model_path_or_zipfile = zipfile.ZipFile(model_path_or_zipfile)

    try:
        data_pickle_bytes = model_path_or_zipfile.read("archive/data.pkl")
        archive_name = "archive"
    except KeyError:
        archive_name = get_archive_name(model_path_or_zipfile, True)
        data_pickle_bytes = model_path_or_zipfile.read(f"{archive_name}/data.pkl")

    def persistent_id_load_fn(arg):
        return arg

    build_tensor = functools.partial(_build_tensor, model_path_or_zipfile, archive_name)
    model = pickle_bytes_safe_load_dict(
        data_pickle_bytes, persistent_id_load_fn,
        reduce_fns_custom = {
            "torch._utils _rebuild_tensor_v2": build_tensor,
        },
        reduce_fns_ignore_unknown = True,
        extended = extended,
    )

    return model


class LazyTensor():
    def load(self, reload = False, dtype = None):
        raise NotImplementedError()

    def dtype(self):
        raise NotImplementedError()

    def unload(self):
        raise NotImplementedError()

    def load_copy(self):
        raise NotImplementedError()


@dataclasses.dataclass()
class TorchLazyTensor(LazyTensor):
    _ctx: Any
    _zip_file: ZipFile
    _archive_name: str
    _storage_tup: tuple
    _data_path: str

    _storage_offset: int
    _size: torch.Size
    _stride: Union[tuple, int]
    _requires_grad: bool

    tensor: Optional[torch.Tensor] = None

    def load(self, reload = False, dtype = None):
        if self.tensor is not None and not reload:
            return self.tensor

        self.tensor = self.load_copy()
        return self.tensor

    def dtype(self):
        return DTYPE_MAP[self._storage_tup[1]][0]

    def unload(self):
        self.tensor = None

    def load_copy(self):
        return _build_tensor(
            self._zip_file, self._archive_name, self._storage_tup, self._storage_offset, self._size,
            self._stride, self._requires_grad, None,
        )

    def load_meta(self):
        return _build_tensor_meta(self._storage_tup, self._storage_offset, self._size, self._stride)


def _build_tensor_meta(storage, storage_offset, size, stride):
    (storage, dtype_str, index, location, element_count) = storage
    if storage != "storage":
        raise ValueError("expected storage", storage)

    dtype, dtype_size = DTYPE_MAP[dtype_str]
    # return torch.empty(size, stride, dtype = dtype, device = "meta")
    tensor = torch.empty_strided(tuple(size), stride, dtype = dtype, device = "meta")
    return tensor


def _build_lazy_tensor(ctx, zipfile: ZipFile, archive_name: str, storage_tup, storage_offset, size, stride, requires_grad, backward_hooks):
    if backward_hooks:
        raise ValueError("unsupported _rebuild_tensor_v2 arg", (storage_offset, stride, backward_hooks))

    (storage, dtype_str, index, location, element_count) = storage_tup
    if storage != "storage":
        raise ValueError("expected storage", storage)

    dtype, dtype_size = DTYPE_MAP[dtype_str]
    data_path = f"{archive_name}/data/{index}"
    data_size = zipfile.getinfo(data_path).file_size

    expected_size = element_count * dtype_size
    if data_size != expected_size:
        raise ValueError("read unexpected amount of bytes",
                         data_size, expected_size, data_path, element_count, dtype_size)

    return TorchLazyTensor(ctx, zipfile, archive_name, storage_tup, data_path, storage_offset, torch.Size(size), stride, requires_grad)


def torch_safe_load_dict_lazy(model_path_or_zipfile: Union[str, ZipFile], extended: bool = False, tensor_ctx = None):
    if isinstance(model_path_or_zipfile, str):
        model_path_or_zipfile = ZipFile(model_path_or_zipfile)

    try:
        data_pickle_bytes = model_path_or_zipfile.read("archive/data.pkl")
        archive_name = "archive"
    except KeyError:
        archive_name = get_archive_name(model_path_or_zipfile, True)
        data_pickle_bytes = model_path_or_zipfile.read(f"{archive_name}/data.pkl")

    def persistent_id_load_fn(arg):
        return arg

    build_tensor = functools.partial(_build_lazy_tensor, tensor_ctx, model_path_or_zipfile, archive_name)
    model = pickle_bytes_safe_load_dict(
        data_pickle_bytes, persistent_id_load_fn,
        reduce_fns_custom = {
            "torch._utils _rebuild_tensor_v2": build_tensor,
        },
        reduce_fns_ignore_unknown = True,
        extended = extended,
    )

    return model
