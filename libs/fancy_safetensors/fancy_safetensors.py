#
# Copyright (c) 2024 LatentRat
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

"""
 Safetensors only allows a dict of tensors and an optional _metadata dict of {str:str}
 for "safety", which is really limiting so let's extend that to storing any of the basic
 python types [str, int, float, bool, list, tuple, dict, set] + Tensors, or any arbitrarily nested form of them.

 format:
 state_dict {key:tensor,...}
 _metadata = {
    "__fancy_safetensors_encoded__": "json_encoded_info_str",
 }

json_encoded_info:
    {
        "encoded_obj": encoded_obj,
    }

encoded_obj:
    The thing that's being saved, any of the types or nested types above.
    Encoding scheme:
        dict: ["dict", [[key, val], ...] # key and val both encoded
        list: ["list", [list items encoded] ] # each item encoded
        tuple: ["tuple", [tuple items encoded] ]
        set: ["set", [set items encoded] ]
        int|float|bool: original value #TODO: check if that works for everything including big/huge ints and re float precision in json
        tensor: ["tensor", "tensor_name"] # tensor name as string to what it's named in the state_dict

Tensors are just given directly as name:tensor dict to safetensors and stored by it.
state_dict:
    Note: Any names for tensor keys in the state_dct are considered fine (as far as the "protocol" and loading goes),
    the following is just what the lib currently chooses for writing.
    By default the state_dict keys for tensors are  "__fancy_tensor.[num]" with continually increasing number.
    Optionally if the tensor is in a dict then the state_dict key used for it can be kept the same, if possible.
    In case of duplicates "[dict_key_name].[num]" with first unused num is used, starting from 0.

Decoding: Undo all that based on state_dict and _metadata.
"""

from __future__ import annotations

import json
from typing import Union

from torch import Tensor

SupportedTypes = Union[str, int, float, bool, list, tuple, dict, set, Tensor]
SupportedTypesTup = (str, int, float, bool, list, tuple, dict, set, Tensor)


def encode(object: SupportedTypes, transform_tensor_fn = None, keep_dict_keys: bool = False) -> tuple[dict[str, Tensor], dict[str, str]]:
    encoder = _Encoder(keep_dict_keys = keep_dict_keys, transform_tensor_fn = transform_tensor_fn)
    encoded_obj = encoder.encode(object)
    state_dict = encoder.finalize_state_dict()

    encoded_info = {
        "encoded_obj": encoded_obj,
    }

    encoded_obj_str = json.dumps(encoded_info)
    _metadata = {
        "__fancy_safetensors_encoded__": encoded_obj_str,
    }
    return state_dict, _metadata


def is_encoded(_metadata: dict[str, str] | None) -> bool:
    if not _metadata:
        return False

    encoded_obj_str = _metadata.get("__fancy_safetensors_encoded__")
    if not encoded_obj_str:
        return False
    return True


def decode(state_dict: dict[str, Tensor], _metadata: dict[str, str],
           return_clean_metadata = False) -> SupportedTypes:
    if not _metadata:
        raise ValueError("metadata required")

    encoded_obj_str = _metadata.get("__fancy_safetensors_encoded__")
    if not encoded_obj_str:
        raise ValueError("metadata not properly formatted")

    encoded_info = json.loads(encoded_obj_str)
    encoded_obj = encoded_info.get("encoded_obj")
    if not encoded_obj:
        raise ValueError("encoded_obj missing")

    obj = _decode(encoded_obj, state_dict)
    if return_clean_metadata:
        _metadata = dict(_metadata)
        _metadata.pop("__fancy_safetensors_encoded__")
        return obj, _metadata
    return obj


def _decode(obj, state_dict: dict[str, Tensor]):
    if obj is None:
        return None

    if isinstance(obj, BasicTypes):
        return obj

    if isinstance(obj, list):
        if len(obj) < 2:
            raise ValueError("invalid encoded list", obj)
        item, arg = obj

        if item == "list":
            if not isinstance(arg, list):
                raise ValueError("list arg not list", arg, obj)
            return [_decode(i, state_dict) for i in arg]

        if item == "tuple":
            if not isinstance(arg, list):
                raise ValueError("tuple arg not list", arg, obj)
            return tuple(_decode(i, state_dict) for i in arg)

        if item == "dict":
            if not isinstance(arg, list):
                raise ValueError("dict arg not list", arg, obj)

            map = { }
            for kv in arg:
                key, val = kv
                key = _decode(key, state_dict)
                val = _decode(val, state_dict)
                if key in map:
                    raise ValueError("duplicate key", key)
                map[key] = val
            return map

        if item == "set":
            if not isinstance(arg, list):
                raise ValueError("set arg not list", arg, obj)
            res = set()
            for i in arg:
                i = _decode(i, state_dict)
                if i in res:
                    raise ValueError("duplicate set item", i)
                res.add(i)
            return res

        if item == "tensor":
            name = arg
            if not isinstance(name, str):
                raise ValueError("tensor name not string", name, obj)
            tensor = state_dict[name]
            if not isinstance(tensor, Tensor):
                raise ValueError("tensor not tensor", tensor, obj)
            return tensor

        raise ValueError("unexpected container item type", item, obj)

    raise ValueError("unexpected object type", type(obj), obj)


BasicTypes = (str, int, float, bool)
BasicContainers = (list, tuple, dict, set)
AllowedTypes = BasicTypes + BasicContainers + (Tensor,)


class _Encoder():
    def __init__(self, keep_dict_keys: bool = False, transform_tensor_fn = None):
        self._state_dict = { }
        self._tensor_nums = { }

        self.keep_dict_keys = keep_dict_keys
        self.transform_tensor_fn = transform_tensor_fn

    def encode(self, object: SupportedTypes, dict_key: str | None = None):
        if object is None:
            return None

        if not isinstance(object, AllowedTypes):
            raise TypeError(f"object type {type(object)} not allowed")

        if isinstance(object, BasicTypes):
            return object

        if isinstance(object, BasicContainers):
            if isinstance(object, list):
                return ["list", [self.encode(i) for i in object]]
            if isinstance(object, tuple):
                return ["tuple", [self.encode(i) for i in object]]
            if isinstance(object, dict):
                return ["dict", [[self.encode(k), self.encode(v, dict_key = k)] for k, v in object.items()]]
            if isinstance(object, set):
                return ["set", [self.encode(i) for i in object]]
            raise ValueError("unexpected container type", type(object))

        if isinstance(object, Tensor):  # TODO: option to support Tensor types from other libs
            if self.transform_tensor_fn is not None:
                object = self.transform_tensor_fn(object)

            if not self.keep_dict_keys or dict_key is None:
                dict_key = "__fancy_tensor"

            use_key = dict_key
            while use_key in self._state_dict:
                use_key = f"{dict_key}.{self._tensor_nums.setdefault(dict_key, 0)}"
                self._tensor_nums[dict_key] += 1

            self._state_dict[use_key] = object
            return ["tensor", use_key]

        raise ValueError("unexpected object type", type(object))

    def finalize_state_dict(self):
        sd = self._state_dict
        self._state_dict = None
        return sd
