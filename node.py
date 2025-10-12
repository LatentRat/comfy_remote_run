# ComfyUI remote node execution nodes.
#    Copyright (C) 2024  LatentRat
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

# TODO: option to actually run IS_CHANGED checks remotely
# TODO: option to lazily serialize/send inputs only when needed remotely? and option to cache them remotely too
# TODO: torch weights_only option
# TODO: clear remote history item afterwards option

import base64
import collections
import copy
import enum
import functools
import gzip
import hashlib
import math
import queue
import re
import threading
import time
import os
import weakref
from io import BytesIO

from uuid import uuid4
from zipfile import ZipFile

import execution
import nodes
from comfy_execution.graph import DynamicPrompt

import requests
import torch
import websockets.sync.client

from .libs.safe_load import torch_safe_load_dict
from .libs.fancy_safetensors import torch as fst_torch

import json

import logging as _logging

from comfy.comfy_types import IO as CoIO

logger = _logging.getLogger(__name__)

if (IS_DEV := os.environ.get("DEV") == "1"):
    IS_DEV_PROMPTSAVE = True
    IS_DEV_PROMPTSAVE = False


    def DEBUG_PROMPT_SAVE(label, prompt):
        from pathlib import Path
        dir = Path(f"./dev/jsons/")
        if not dir.exists():
            dir.mkdir(parents = False, exist_ok = True)

        from comfy.cli_args import args
        import datetime
        now = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d_%H-%M-%S.%f")
        Path(f"./dev/jsons/port={args.port}____{label}__{now}__{time.time_ns()}.json").write_text(json.dumps(prompt))

jdumps = lambda data: json.dumps(data).replace("\n", "")

_NUM_OUTPUTS = int(os.environ.get("RAT_REMOTE_RUN_NUM_OUTPUTS") or 10)
logger.info("%s", f"RemoteRun nodes using _NUM_OUTPUTS={_NUM_OUTPUTS}")

TOGGLE_CHOICES = ("only_locally", "only_remotely")
INPUTS_OFF_OPTIONS = ["lazy", "disconnected"]
BLOCK_OPTIONS = ["error", "block_execution_silent", "block_execution_verbose", "passthrough", "return_none"]
SERIALIZATION_OPTIONS = ["safe_torch_pt", "unsafe_torch_pt", "fancy_safetensors"]


class BinaryResponseMessageID(enum.IntEnum):
    BINARY_RESPONSE_MESSAGE_ID = 12341119
    GZIPPED_BINARY_RESPONSE_MESSAGE_ID = 12341120


def serialization_load_input(name = "serialization"):
    return { name: (SERIALIZATION_OPTIONS, { "default": "fancy_safetensors" }) }


def response_input(name = "response"):
    return { name: (["base64_result", "binary"], { "default": "binary" }) }


def ws_settings_inputs():
    return {
        "ws_compression":            ("BOOLEAN", { "default": False }),
        # strings to make it easier to add more in case of wanting a partial option without breaking backwards compatibility
        "forward_progress_messages": (["on", "off"], { "default": "on" }),
    }


def shared_input_output_settings():
    return {
        "max_size_mb":      ("INT", { "default": 32, "min": 0, "max": 8 * 1024, "step": 1 }),
        "gzip_compression": ("BOOLEAN", { "default": False }),
        "gzip_level":       ("INT", { "default": 9, "min": 1, "max": 9, "step": 1 }),
    }


def lazy_input_settings():
    return {
        "lazily_transfer":             ("BOOLEAN", { "default": False }),
        "skip_lazy_transfer_under_mb": ("FLOAT", {
            "default": 0.25, "min": 0.0, "max": 10 * 1024.0, "step": 0.1,
            "tooltip": "Even when lazily transferring inputs is enabled still always transfer inputs smaller than this size in MB directly.",
        }),
    }


class RemoteRunSerializerOutNode():
    TYPE_NAME = "RAT_RemoteRunSerializerOut"
    DISPLAY_NAME = "RemRun Serializer Output (Internal)"

    CATEGORY = "Remote Run/__Internal__/"
    FUNCTION = "run"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, is_changed: bool = False, **_kwargs):
        return time.time_ns() if is_changed else None

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                **serialization_load_input(),
            },
            "optional": {
                **dict((f"input_{num}", (CoIO.ANY, { })) for num in range(_NUM_OUTPUTS)),
                **shared_input_output_settings(),
                **response_input(),
                "is_changed": ("BOOLEAN", { "default": False }),
            },
        }
        return inputs

    RETURN_TYPES = ("STRING",)

    def run(self,
            serialization: str,
            max_size_mb: int | None = None, gzip_compression: bool = False, gzip_level: int = 9, response = None,
            **kwargs,
            ):
        inputs = { k: v for k, v in kwargs.items() if k.startswith("input_") }

        data = serialize_obj(serialization, inputs)
        if response == "binary":
            import server
            instance = server.PromptServer.instance
            client_id = instance.client_id
            if not client_id:
                raise ValueError("No client connected to send binary response to")

            data_type = BinaryResponseMessageID.BINARY_RESPONSE_MESSAGE_ID.value
            if gzip_compression:
                start = time.monotonic_ns()
                comp = gzip.compress(data, compresslevel = gzip_level)
                end = time.monotonic_ns()
                took_ns = (end - start)
                mb_s = (len(data) * 1e9) / (took_ns * 1024 * 1024)
                took_sec = took_ns / 1e9
                factor = len(comp) / len(data)
                if factor < 0.99:
                    logger.info("%s", f"RemoteRunSerializerOutNode.run serialization={serialization!r} {response=!r} "
                                      f"gzip compressed {len(data)} to {len(comp)} bytes, factor {factor:.3f}, "
                                      f"took {took_sec:.3f} sec, {mb_s:.2f} MB/s")
                    data = comp
                    data_type = BinaryResponseMessageID.GZIPPED_BINARY_RESPONSE_MESSAGE_ID.value
                else:
                    logger.info("%s", f"RemoteRunSerializerOutNode.run serialization={serialization!r} {response=!r} "
                                      f"gzip compression not effective, kept original {len(data)} bytes "
                                      f"over compressed {len(comp)} bytes, factor {factor:.3f}, "
                                      f"took {took_sec:.3f} sec, {mb_s:.2f} MB/s")

            logger.info("%s", f"RemoteRunSerializerOutNode.run serialization={serialization!r} {response=!r} "
                              f"data_size={len(data)}, {max_size_mb=} {client_id=!r}")

            if max_size_mb and len(data) > max_size_mb * 1024 * 1024:
                raise ValueError(f"Serialized data too large: {len(data)} bytes, max_size_mb={max_size_mb}")

            instance.send_sync(data_type, data, client_id)
            return ("",)

        results = base64.b64encode(data).decode()
        logger.info("%s", f"RemoteRunSerializerOutNode.run serialization={serialization!r} {response=!r} data_size={len(data)}, "
                          f"b64encode_size={len(results)}, {max_size_mb=}")
        if max_size_mb and len(results) > max_size_mb * 1024 * 1024:
            raise ValueError(f"Serialized b64 data too large: {len(results)} bytes, max_size_mb={max_size_mb}")

        results = (results,)
        return {
            "results": results,
            "ui":      { "results": results }
        }


def setup_remote_run_api_route(expected_set_up: bool):
    import server
    from aiohttp import web

    ins = server.PromptServer.instance
    url = "/remote_run/data/"
    cur_routes = [ins.routes[i] for i in range(len(ins.routes))]
    routes = [i for i in cur_routes if i.path == url and (i.method or "").upper() == "POST"]
    if routes:
        if len(routes) > 1:
            raise ValueError("Multiple routes for remote run data endpoint???", routes)
        route_def = routes[0]
        handler = route_def.handler
        data_requests = handler._data_requests
    else:
        if expected_set_up:
            raise ValueError("No existing route for remote run data endpoint, disabled or server startup changed")

        data_requests = weakref.WeakValueDictionary()

        @ins.routes.post(url)
        async def receive_data(request):
            key = request.query.get("key")
            status = request.query.get("status")
            logger.info("%s", f"RemoteRunDeserializerOutNode.receive_data /remote_run/data/ POST handler key={key!r} status={status!r}")
            if not key:
                return web.Response(status = 400, text = "Missing key")
            notif = data_requests.get(key)
            if notif is None:
                return web.Response(status = 400, text = "No data requested for this key")

            if status != "data":
                notif.update("remote_error", f"remote side error status: {status!r}")
                return web.Response(status = 200, text = f"Error status: {status!r}")

            notif.update("started", None)
            try:
                data = await request.read()
                notif.update("data", data)
            except Exception as ex:
                notif.update("error", ex)
            return web.Response(status = 200, text = "OK")

        # ins.app.add_routes(api_routes)
        receive_data._data_requests = data_requests

    if len(data_requests) > 5:
        logger.warning("%s", f"RemoteRunDeserializerOutNode.setup_server has {len(data_requests)} data requests pending, shouldn't happen")
    return ins, data_requests


class RemoteRunDeserializerOutNode():
    TYPE_NAME = "RAT_RemoteRunDeserializerOut"
    DISPLAY_NAME = "RemRun Deserializer Output (Internal)"

    CATEGORY = "Remote Run/__Internal__/"
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        inputs = { }
        for num in range(_NUM_OUTPUTS):
            inputs[f"input_{num}_info"] = (CoIO.ANY, { "lazy": True })
            inputs[f"input_{num}_data"] = (CoIO.ANY, { "lazy": True })
        return {
            "required": {
            },
            "optional": {
                **inputs,
                "total_timeout":   ("FLOAT", { }),
                "request_timeout": ("FLOAT", { }),
                "is_changed":      ("BOOLEAN", { "default": False }),
            },
        }

    RETURN_TYPES = tuple([CoIO.ANY] * _NUM_OUTPUTS)

    def run(self, total_timeout: float = None, request_timeout: float = None, **kwargs):
        get_timeout = _make_adjusted_timeout(total_timeout, allow_empty = True)

        info_keys = [i for i in kwargs.keys() if i.startswith("input_") and i.endswith("_info")]
        nums = sorted(int(i[6:-5]) for i in info_keys)
        outputs = { }
        for in_num in nums:
            info_str = kwargs[f"input_{in_num}_info"]
            data = kwargs[f"input_{in_num}_data"]
            if info_str == "INPUT":
                outputs[in_num] = data
                continue

            info = json.loads(info_str)
            data_type = info["type"]
            if data_type == "serialized":
                outputs[in_num] = deserialize_response(info["serialization"], data, info.get("compression"))
                continue

            if data_type == "request":
                key = info["key"]
                serialization = info["serialization"]
                if serialization not in SERIALIZATION_OPTIONS:
                    raise ValueError("Invalid serialization option", serialization, SERIALIZATION_OPTIONS, info)

                server_ins, weak_data_requests = setup_remote_run_api_route(expected_set_up = True)
                client_id = server_ins.client_id
                if not client_id:
                    raise ValueError("No client connected to request data from")

                class Notif():
                    def __init__(self):
                        self.resq = queue.Queue()

                    def update(self, status, data):
                        self.resq.put((status, data))

                    def cleanup(self):
                        _res = weak_data_requests.pop(key, None)
                        # print("data_pop", key, _res)

                notif = Notif()
                try:
                    weak_data_requests[key] = notif
                    logger.info("%s", f"RemoteRunDeserializerOutNode.run sending NEED_DATA data key={key!r}")
                    server_ins.send_sync("NEED_DATA", dict(key = key), client_id)
                    status, _data = notif.resq.get(timeout = get_timeout(request_timeout))

                    if status == "remote_error":
                        raise ValueError("remote side error requesting data", _data, key, info)
                    if status != "started":
                        raise ValueError("unexpected status waiting for data", status, key, info, _data)

                    result, data = notif.resq.get(timeout = get_timeout(request_timeout))
                    logger.info("%s", f"RemoteRunDeserializerOutNode.run received data key={key!r} {data and len(data)} bytes")
                finally:
                    notif.cleanup()

                if result != "data":
                    raise ValueError("error receiving data", result, key, info, data)

                deserialized = deserialize_response(serialization, data, info.get("compression"))
                outputs[in_num] = deserialized
                continue

            raise ValueError("Invalid input data type", in_num, info)

        return tuple(outputs.get(i, None) for i in range(_NUM_OUTPUTS))


class RemoteRunTogglerNode():
    TYPE_NAME = "RAT_RemoteRunToggler"
    DISPLAY_NAME = "RemRun Toggle"

    """
    This is a Input Toggle Switch that can be used to enable/disable certain parts of a graph
    in the exact opposite way between the local and remote side using lazy evaluation or prompt preprocessing.

    If the switch run_side is set to enabled it will run all inputs, otherwise set them all to off/lazy
    or disconnected before prompt execution, depending on [inputs_when_off] setting.
    When a RemoteRun runner node encounters one of these switches in a prompt it just toggles
    the run_side value so the remote side will do the exact opposite of the local one,
    without even knowing it's the remote side.
    """
    CATEGORY = "Remote Run"
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "enabled":         (TOGGLE_CHOICES, { }),
                "inputs_when_off": (INPUTS_OFF_OPTIONS, { "default": "lazy" }),
                "when_off":        (BLOCK_OPTIONS, { "default": "passthrough" }),
            },
            "optional": {
                **dict((f"input_{num}", (CoIO.ANY, { "lazy": True })) for num in range(_NUM_OUTPUTS)),
                "_ignore_": ("INTERNAL", { "default": "" }),
            },
            # "hidden":   { "own_id": "UNIQUE_ID" },
            # "hidden":   { "prompt": "PROMPT" },
            "hidden":   {
                "own_id": "UNIQUE_ID",
                # "prompt": "PROMPT",
            }
        }
        return inputs

    RETURN_TYPES = tuple([CoIO.ANY] * _NUM_OUTPUTS)

    @classmethod
    def IS_CHANGED(self, **_kwargs):
        return None

    def check_lazy_status(self, enabled: str, own_id = None, **kwargs):
        if enabled not in TOGGLE_CHOICES:
            raise ValueError("Invalid enabled value", own_id, enabled, TOGGLE_CHOICES)

        if enabled == TOGGLE_CHOICES[0]:
            inputs_with_connections = [i for i in kwargs.keys() if i.startswith("input_")]
            # print("RemoteRunToggler.check_lazy_status", enabled, own_id, inputs_with_connections, "all:", { k: (v is not None) for k, v in kwargs.items() })
            return inputs_with_connections

        # print("RemoteRunToggler.check_lazy_status", enabled, own_id, [], "all:", { k: (v is not None) for k, v in kwargs.items() })
        return []

    def run(self, enabled: str, when_off: str, _ignore_ = None, own_id: str = None, **kwargs):
        disconnected_input_ids = _ignore_
        # print(own_id, enabled, _ignore_)
        print(f"{self.__class__.__name__}.run {own_id=} {enabled=} {when_off=} {disconnected_input_ids=}")
        # print("RemoteRunToggler.prompt", prompt)
        if enabled == TOGGLE_CHOICES[1]:
            error_msg = f"{self.DISPLAY_NAME!r} node output is off and set to 'error'"
            return block_option_return(when_off or "passthrough", kwargs, _NUM_OUTPUTS, error_msg, "RemoteRunToggler")

        return tuple(kwargs.get(f"input_{num}", None) for num in range(_NUM_OUTPUTS))


def _shared_input_types():
    return {
        "remote_url":      ("STRING", {
            "default": "http://127.0.0.1:8189/",
            "tooltip": "ComfyUI instance URL",
        }),
        "total_timeout":   ("FLOAT", {
            "tooltip": "Timeout for everything total (sending prompt and running) in seconds. 0 to disable.",
            "default": 90.0, "min": 0, "max": 24 * 3600.0, "step": 0.1
        }),
        "request_timeout": ("FLOAT", {
            "tooltip": "Prompt & data post timeout in seconds",
            "default": 45.0, "min": 0.1, "max": 24 * 3600.0, "step": 0.1
        }),
        **serialization_load_input(),
        # **output_nodes_input(),
    }


def _get_max_size(max_size_mb: int | None):
    if not max_size_mb or max_size_mb <= 0:
        return None, None

    max_bytes = max_size_mb * 1024 * 1024
    ws = math.ceil(max_bytes * 1.1)

    return max_bytes, ws


class RemoteRunInputNode():
    TYPE_NAME = "RAT_RemoteRunInput"
    DISPLAY_NAME = "RemRun Input Graph(s)"

    NUM_OUTPUTS = _NUM_OUTPUTS
    CATEGORY = "Remote Run"
    FUNCTION = "run"

    @classmethod
    def IS_CHANGED(cls, is_changed: bool = False, **_kwargs):
        return time.time_ns() if is_changed else None

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                **_shared_input_types(),
            },
            "optional": {
                **dict((f"input_{num}", (CoIO.ANY, { "lazy": True })) for num in range(_NUM_OUTPUTS)),
                **shared_input_output_settings(),
                **lazy_input_settings(),
                **response_input(),
                **ws_settings_inputs(),
                "run_outputs_connected_to_inputs": (["ignore", "run"], { "default": "ignore" }),
                "inputs_when_local":               (INPUTS_OFF_OPTIONS, { "default": "disconnected" }),
                "is_changed":                      ("BOOLEAN", { "default": False }),
                "_ignore_":                        ("INTERNAL", { "default": "" }),
            },
            "hidden":   {
                "dynprompt": "DYNPROMPT",
                "own_id":    "UNIQUE_ID",
            }
        }
        return inputs

    RETURN_TYPES = tuple([CoIO.ANY] * _NUM_OUTPUTS)

    def check_lazy_status(self, **_kwargs):
        return []

    def run(self,
            remote_url: str, total_timeout: float, request_timeout: float, serialization: str,
            max_size_mb: int | None = None, gzip_compression: bool = False, gzip_level: int = 9, response = None,
            run_outputs_connected_to_inputs = "disconnected", is_changed: bool = False,
            ws_compression: bool = False, forward_progress_messages: str = "off",
            lazily_transfer: bool = False, skip_lazy_transfer_under_mb: float = 0.25,
            dynprompt: DynamicPrompt = None, own_id = None, _ignore_ = None,
            **_kwargs,
            ):
        prompt = { i: dynprompt.get_node(i) for i in dynprompt.all_node_ids() }
        return partial_json_expansion(
            prompt, own_id, remote_url, serialization,
            total_timeout, request_timeout,
            max_size_mb, gzip_compression, gzip_level,
            ws_compression, forward_progress_messages, is_changed, response,
            lazily_transfer, skip_lazy_transfer_under_mb,
            run_outputs_connected_to_inputs,
        )


class RemoteRunInputOutputNode(RemoteRunInputNode):
    TYPE_NAME = "RAT_RemoteRunInputOutput"
    DISPLAY_NAME = "RemRun Input Graph(s) Output"
    OUTPUT_NODE = True


class RemoteRunJsonNode():
    TYPE_NAME = "RAT_RemoteRunJson"
    DISPLAY_NAME = "RemRun JSON"

    NUM_OUTPUTS = _NUM_OUTPUTS
    CATEGORY = "Remote Run"
    FUNCTION = "run"

    @classmethod
    def IS_CHANGED(cls, is_changed: bool = False, **_kwargs):
        return time.time_ns() if is_changed else None

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                **_shared_input_types(),
                "JSON": ("STRING", { }),
            },
            "optional": {
                **dict((f"input_{num}", (CoIO.ANY, { })) for num in range(_NUM_OUTPUTS)),
                **shared_input_output_settings(),
                **lazy_input_settings(),
                **response_input(),
                **ws_settings_inputs(),
                "is_changed": ("BOOLEAN", { "default": False }),
            },
        }
        return inputs

    RETURN_TYPES = tuple([CoIO.ANY] * _NUM_OUTPUTS)

    def run(self,
            remote_url: str, total_timeout: float, request_timeout: float, JSON: str, serialization: str,
            max_size_mb: int | None = None, gzip_compression: bool = False, gzip_level: int = 9, response = None,
            ws_compression: bool = False, forward_progress_messages: str = "off",
            lazily_transfer: bool = False, skip_lazy_transfer_under_mb: float = 0.25,
            **kwargs,
            ):
        max_size_mb = max_size_mb or None
        run_obj = json.loads(JSON)
        extra_output_ids = run_obj.get("extra_output_ids") or None
        run_prompt = run_obj["prompt"]
        data_total = 0

        lazy_data = { }
        skip_lazy_transfer_under_mb = skip_lazy_transfer_under_mb * 1024 ** 2 if skip_lazy_transfer_under_mb else None

        for node_id, node in run_prompt.items():
            # serialize needed inputs to this node and set up
            # Deserializer data output nodes with the serialized data in the remote prompt
            if node["class_type"] == RemoteRunDeserializerOutNode.TYPE_NAME:
                config = node.pop("deserializer_config")
                outputs = config["outputs"]
                other_inputs = config["other_inputs"]
                node_lazily_transfer = other_inputs.get("lazily_transfer", None)
                if node_lazily_transfer is not None and node_lazily_transfer != "from_input_node":
                    node_lazily_transfer = { "on": True, "off": False }[node_lazily_transfer.lower()]
                    node_skip_under = other_inputs["skip_lazy_transfer_under_mb"]
                else:
                    node_lazily_transfer = lazily_transfer
                    node_skip_under = skip_lazy_transfer_under_mb

                inputs = { }
                for output_name, output_data in outputs.items():
                    if not isinstance(output_data, list) or not len(output_data) == 2:
                        raise ValueError("Invalid output data", output_data, config)

                    if not output_name.startswith("output_") and output_name[7:].isdigit():
                        raise ValueError("Invalid output data", output_data, config)
                    output_num = int(output_name[7:])

                    output_data_type, output_data_value = output_data
                    if output_data_type == "RunJsonInput":
                        if not output_data_value.startswith("input_") and output_data_value[6:].isdigit():
                            raise ValueError("Invalid output data", output_data, config)

                        local_input_name = output_data_value
                        input = kwargs[local_input_name]

                        # TODO: skip serialization if input is basic python types
                        input_data = serialize_obj(serialization, input)
                        compression = None
                        if gzip_compression:
                            start = time.monotonic_ns()
                            comp = gzip.compress(input_data, compresslevel = gzip_level)
                            end = time.monotonic_ns()
                            took_ns = (end - start)
                            mb_s = (len(input_data) * 1e9) / (took_ns * 1024 * 1024)
                            took_sec = took_ns / 1e9
                            factor = len(comp) / len(input_data)
                            if factor < 0.99:
                                logger.info("%s", f"RemoteRunJsonNode.run serialization={serialization!r} {response=!r} "
                                                  f"gzip compressed {len(input_data)} to {len(comp)} bytes, factor {factor:.3f}, "
                                                  f"took {took_sec:.3f} sec, {mb_s:.2f} MB/s")
                                input_data = comp
                                compression = "gzip"
                            else:
                                logger.info("%s", f"RemoteRunJsonNode.run serialization={serialization!r} {response=!r} "
                                                  f"gzip compression not effective, kept original {len(input_data)} bytes "
                                                  f"over compressed {len(comp)} bytes, factor {factor:.3f}, "
                                                  f"took {took_sec:.3f} sec, {mb_s:.2f} MB/s")

                        if node_lazily_transfer and node_skip_under and len(input_data) < node_skip_under:
                            node_lazily_transfer = False

                        if node_lazily_transfer:
                            data_total += len(input_data)
                            if max_size_mb and data_total > max_size_mb * 1024 * 1024:
                                raise ValueError(f"Serialized data too large: {data_total} bytes, max_size_mb={max_size_mb}")

                            key = hashlib.sha256(hashlib.sha256(input_data).digest()).hexdigest()
                            lazy_data[key] = input_data

                            inputs[f"input_{output_num}_info"] = json.dumps({
                                "type":          "request",
                                "key":           key,
                                "serialization": serialization,
                                "compression":   compression,
                            })
                            inputs[f"input_{output_num}_data"] = True
                        else:
                            input_data_b64 = base64.b64encode(input_data).decode()
                            data_total += len(input_data_b64)
                            if max_size_mb and data_total > max_size_mb * 1024 * 1024:
                                raise ValueError(f"Serialized data too large: {data_total} bytes, max_size_mb={max_size_mb}")

                            inputs[f"input_{output_num}_info"] = json.dumps({
                                "type":          "serialized",
                                "serialization": serialization,
                                "compression":   compression,
                            })
                            inputs[f"input_{output_num}_data"] = input_data_b64
                    elif output_data_type == "CONSTANT":
                        inputs[f"input_{output_num}_info"] = "INPUT"
                        inputs[f"input_{output_num}_data"] = output_data_value
                    else:
                        raise ValueError("Invalid output data", output_data, config)

                node["inputs"] = {
                    "serialization": serialization,
                    **inputs,
                }

        serializers = { k: v for k, v in run_prompt.items() if v["class_type"] == RemoteRunSerializerOutNode.TYPE_NAME }
        if len(serializers) != 1:
            raise ValueError(f"expected exactly one {RemoteRunSerializerOutNode.TYPE_NAME!r} serializer node, got {len(serializers)}",
                             serializers, run_prompt)
        serialized_id = list(serializers.keys())[0]

        post_max, ws_max = _get_max_size(max_size_mb)
        binary_response = response == "binary"
        raw_result = remote_execute_prompt(
            remote_url, run_prompt, serialized_id, extra_output_ids = extra_output_ids,
            ws_compression = ws_compression, forward_progress_messages = forward_progress_messages,
            binary_response = binary_response,
            ws_max_size = ws_max, post_max_size = post_max,
            total_timeout = total_timeout, short_timeout = request_timeout,
            lazy_data = lazy_data,
        )
        obj = deserialize_response(serialization, raw_result)

        result = tuple(obj.get(f"input_{num}", None) for num in range(_NUM_OUTPUTS))
        return result


def block_option_return(option: str, kwargs, num_outputs, error_msg, block_msg):
    option = option.lower()
    if option == "passthrough":
        return tuple(kwargs.get(f"input_{num}", None) for num in range(num_outputs))
    elif option == "block_execution_silent":
        return tuple(execution.ExecutionBlocker(None) for _ in range(num_outputs))
    elif option == "block_execution_verbose":
        return tuple(execution.ExecutionBlocker(f"{block_msg} out={num}") for num in range(num_outputs))
    elif option == "error":
        raise ValueError(error_msg)
    elif option == "return_none":
        return tuple(None for _ in range(num_outputs))
    else:
        raise ValueError(f"Invalid option value: {option!r}")


class RemoteRunStartNode():
    TYPE_NAME = "RAT_RemoteRunStart"
    DISPLAY_NAME = "RemRun Start From Here ->"

    CATEGORY = "Remote Run"
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "optional": {
                **dict((f"input_{num}", (CoIO.ANY, { })) for num in range(_NUM_OUTPUTS)),
                "remote_run_dependent_outputs": (["ignore", "run"], { "default": "ignore" }),
                "inputs_when_local":            (INPUTS_OFF_OPTIONS, { "default": "disconnected" }),
                "outputs_when_local":           (BLOCK_OPTIONS, { "default": "error" }),

                "lazily_transfer":              (["from_input_node", "on", "off"], { "default": "from_input_node" }),
                **{ k: v for k, v in lazy_input_settings().items() if k == "skip_lazy_transfer_under_mb" },
            },
        }
        return inputs

    @classmethod
    def VALIDATE_INPUTS(cls, outputs_when_local: str = None, remote_run_dependent_outputs: str = None, *_args, **_kwargs):
        outputs_when_local = outputs_when_local or "error"
        if outputs_when_local.lower() not in BLOCK_OPTIONS:
            return f"Invalid outputs_when_local value: {outputs_when_local!r}"

        remote_run_dependent_outputs = remote_run_dependent_outputs or "ignore"
        if remote_run_dependent_outputs.lower() not in ("ignore", "run"):
            return f"Invalid remote_run_dependent_outputs value: {remote_run_dependent_outputs!r}"

        return True

    RETURN_TYPES = tuple([CoIO.ANY] * _NUM_OUTPUTS)

    def run(self, outputs_when_local: str = None, **kwargs):
        error_msg = f"{self.DISPLAY_NAME!r} node output used locally, but outputs_when_local is set to 'error'"
        return block_option_return(outputs_when_local or "error", kwargs, _NUM_OUTPUTS, error_msg = error_msg, block_msg = "RemoteRunStartNode")


def _max_id(prompt):
    def toint(i):
        try:
            return int(i)
        except ValueError:
            return 0

    if not len(prompt):
        return 0

    return max(toint(i) for i in prompt.keys())


def make_counter(start: int, map = None):
    def get():
        nonlocal start
        ret = start
        start += 1
        return map(ret) if map is not None else ret

    return get


def get_input_graph_nodes(prompt: dict, root_node_id: str) -> tuple[set[str], set[str]]:
    start_nodes = set()
    input_nodes = set()

    def _nodes(node_id: str, seen):
        if node_id in seen:
            return
        seen.add(node_id)

        node = prompt[node_id]
        if node.get("class_type") == RemoteRunStartNode.TYPE_NAME:
            start_nodes.add(node_id)
            return

        inputs = node["inputs"]
        for inp in inputs.values():
            if isinstance(inp, list):
                input_nodes.add(inp[0])
                _nodes(inp[0], seen)

    _nodes(root_node_id, set())

    return start_nodes, input_nodes


def get_full_dependents_of_outputs(prompt: dict, of_main_node_ids: set[str], stop_at_node_ids = None) -> dict[str, set[str]]:
    # For each main_node_id get all nodes that are directly or indirectly linked to its outputs
    # but stop going down the graph when hitting another stop_at_node_ids (or main_node_id if stop_at_node_ids not given).
    of_main_node_ids = set(of_main_node_ids)
    stop_at_node_ids = set(stop_at_node_ids if stop_at_node_ids is not None else of_main_node_ids)

    inputs_map = { }
    for nid, node in prompt.items():
        inputs = set()
        for v in node["inputs"].values():
            if isinstance(v, list) and len(v) == 2:
                inputs.add(v[0])
        inputs_map[nid] = inputs

    @functools.cache
    def get_all_parents(nid):
        if nid in stop_at_node_ids:
            return set()

        parents = set()
        for inp in inputs_map.get(nid) or []:
            parents.add(inp)
            parents.update(get_all_parents(inp))
        return parents

    res = { i: set() for i in of_main_node_ids }
    for nid in prompt.keys():
        all_parents_until_parent_node_id_or_end = get_all_parents(nid)
        important_parents = all_parents_until_parent_node_id_or_end & of_main_node_ids
        for p in important_parents:
            res[p].add(nid)

    return res


def get_extra_output_ids_dependent_on(prompt: dict, node_ids: set[str], stop_at_node_ids = None):
    # get all nodes that are directly or indirectly linked to the outputs of the given node_ids
    # (generally all RemoteRunStartNodes), stop going down the chain when hitting another remote_run
    # start or input/json node.

    node_ids = set(node_ids)
    stop_at_node_ids = set(stop_at_node_ids if stop_at_node_ids is not None else node_ids)
    deps = get_full_dependents_of_outputs(prompt, node_ids, stop_at_node_ids)

    extra_output_ids = set()
    for parent_id, all_deps in deps.items():
        if not all_deps or parent_id not in node_ids:
            continue
        extra_output_ids.update(all_deps)

    return sorted(extra_output_ids)


def partial_json_expansion(
        prompt: dict, root_node_id: str, remote_url: str, serialization: str,
        total_timeout: float, request_timeout: float,
        max_size_mb: int | None, gzip_compression: bool, gzip_level: int,
        ws_compression: bool, forward_progress_messages: str, is_changed: bool, response: str | None,
        lazily_transfer: bool, skip_lazy_transfer_under_mb: float,
        run_outputs_connected_to_inputs: str,
):
    """
        When there are RemoteStart nodes then go from:
            A -> B -> C -> RemoteStart -> D -> E -> F -> RemoteRunInput -> G -> H -> I
            to
            A -> B -> C -> RemoteRunJson(run_remotely="D -> E -> F") -> G -> H -> I

            Build an expanded graph to replace the RemoteRunInput node with a RemoteRunJson node copying
            the whole input graph going to the RemoteRunInput node up until RemoteStart nodes are hit,
            and replace all those nodes with a RemoteRunJson node that will run that part as prompt remotely instead.
            The RemoteRunJson gets linked to all the inputs of the RemoteStart nodes if any so it can serialize
            them and send them to the remote instance as part of the remote prompt.
            Optionally if remote_run_dependent_outputs==run then also include all other nodes that are connected to
            the outputs of any input graph nodes but don't lead to the RemoteRunInput node and run outputs in those too,
            see below for example, this can be set either for everything if the RemoteRunInput node has it set
            or per RemoteRunStart node for anything going off a node in between that and the RemoteRunInput node.

        If there are no RemoteStart nodes then take all the nodes going into RemoteRunInput.
        Optionally also all nodes that go off those inputs nodes too if run_outputs_connected_to_inputs is set to "run",
        so if there are any outputs as part of the input graph but that doesn't lead to RemoteRunInput those get run too.
        So from:
                A -> B -> C -> D -> RemoteRunInput -> G -> H -> I
                A -> SaveImage[F]
            to:
                RemoteRunJson(run_remotely="A -> B -> C -> D") -> G -> H -> I
            or with run_outputs_connected_to_inputs = "run" to:
                RemoteRunJson(run_remotely="A -> SaveImage[F]; A -> B -> C -> D", run_extra_outputs=["F"]) -> G -> H -> I

    """
    remote_prompt = copy.deepcopy(prompt)
    next_node_id = make_counter(_max_id(remote_prompt) + 1, str)
    rcon, dscon = update_toggles_inplace(remote_prompt, True, True, "remote_run_input_toggle", False, True)

    add_outputs_of_inputs = (run_outputs_connected_to_inputs or "").lower() == "run"

    if rcon or dscon:
        logger.info("%s", f"partial_json_expansion {root_node_id=} toggled {rcon + dscon} nodes, "
                          f"reconnected {rcon},  disconnected {dscon}")

    extra_output_ids = None
    local_expanded_inputs = { }
    remote_run_dependent_outputs_of_nodes = []
    start_node_ids, other_node_ids = get_input_graph_nodes(remote_prompt, root_node_id)

    # The D -> E -> F part(s), anything before the RemoteRunInput node back up to any RemoteRunStart nodes
    # and anything starting on its own.
    small_prompt = { i: remote_prompt[i] for i in (start_node_ids | other_node_ids | { root_node_id }) }
    if start_node_ids:
        # For the remote prompt: just replace each RemoteStart node with a Deserializer node that outputs
        # the serialized and sent over inputs.
        # On the local side as there can be multiple RemoteStart nodes all the inputs to the RemoteStart nodes have to also
        # be linked up to the newly created/expanded RemoteRunJson node and kept track of which serialized inputs goes to which
        # Deserializer node output on the remote side that replace the RemoteStart nodes in the remote prompts.

        next_json_input_num = make_counter(0)

        # On remote prompt change all RemoteRunStart nodes to Deserializer nodes
        for start_id in start_node_ids:
            # TODO: ignore inputs for outputs that aren't used, currently serialized/sent for nothing

            start_node = small_prompt[start_id]
            start_inputs = start_node["inputs"]
            node_remote_run_dependent_outputs = start_inputs.get("remote_run_dependent_outputs")
            if (node_remote_run_dependent_outputs or "").lower() == "run":
                remote_run_dependent_outputs_of_nodes.append(start_id)

            des = { }
            other_inputs = { }
            for input_name, input_value in start_inputs.items():
                if not input_name.startswith("input_"):
                    other_inputs[input_name] = input_value
                    continue

                output_name = "output_" + input_name[6:]
                if isinstance(input_value, list):
                    local_input_name = f"input_{next_json_input_num()}"
                    local_expanded_inputs[local_input_name] = input_value
                    des[output_name] = ("RunJsonInput", local_input_name)
                else:
                    des[output_name] = ("CONSTANT", input_value)

            # The deserializer_config part will be removed by the RemoteRunJson node and rewritten
            # to use the serialized inputs as given.
            small_prompt[start_id] = {
                "class_type":          RemoteRunDeserializerOutNode.TYPE_NAME,
                "deserializer_config": {
                    "outputs":      des,
                    "other_inputs": other_inputs,
                },
            }

    if add_outputs_of_inputs:
        input_node_ids = start_node_ids | other_node_ids
        stop_at = start_node_ids | { root_node_id }
        extra_output_ids = get_extra_output_ids_dependent_on(remote_prompt, input_node_ids, stop_at_node_ids = stop_at)
        logger.info("partial_json_expansion %s: add_outputs_of_inputs -> extra_output_ids=%s",
                    root_node_id, (len(extra_output_ids), extra_output_ids))
    elif remote_run_dependent_outputs_of_nodes:
        # Add all output nodes that go off away from the input graph, so between a start or StartNode to RemoteRunInput node,
        # but don't themselves lead to the RemoteRunInput node.
        extra_output_ids = get_extra_output_ids_dependent_on(remote_prompt, set(remote_run_dependent_outputs_of_nodes), stop_at_node_ids = { root_node_id })
        logger.info("partial_json_expansion %s: remote_run_dependent_outputs_of_nodes=%s -> extra_output_ids=%s",
                    root_node_id, remote_run_dependent_outputs_of_nodes, (len(extra_output_ids), extra_output_ids))

    # This is really all dependent nodes not just Output ones but no need to limit to only output ones since any normal non output
    # nodes given will just be ignored by comfy prompt handling (currently).
    extra_output_ids = list(extra_output_ids) if extra_output_ids else None
    for i in extra_output_ids or []:
        if i not in small_prompt:
            small_prompt[i] = remote_prompt[i]

    remote_prompt = small_prompt
    root_node = remote_prompt.pop(root_node_id)
    root_inputs = { k: v for k, v in root_node["inputs"].items() if isinstance(v, list) }
    remote_prompt[root_node_id] = {
        "class_type": RemoteRunSerializerOutNode.TYPE_NAME,
        "inputs":     {
            "serialization": serialization,
            "max_size_mb":   max_size_mb,
            "is_changed":    is_changed,
            **root_inputs,
        },
    }

    DEBUG_PROMPT_SAVE("exported_remote_prompt", remote_prompt) if IS_DEV and IS_DEV_PROMPTSAVE else None

    remote_obj = dict(
        prompt = remote_prompt,
        extra_output_ids = extra_output_ids,
    )
    json_node_id_str = next_node_id()
    new_graph = {
        json_node_id_str: {
            "class_type": RemoteRunJsonNode.TYPE_NAME,
            "inputs":     dict(
                JSON = json.dumps(remote_obj),
                remote_url = remote_url,
                total_timeout = total_timeout,
                request_timeout = request_timeout,
                serialization = serialization,
                max_size_mb = max_size_mb,
                gzip_compression = gzip_compression,
                gzip_level = gzip_level,
                ws_compression = ws_compression,
                forward_progress_messages = forward_progress_messages,
                is_changed = is_changed,
                response = response,
                lazily_transfer = lazily_transfer,
                skip_lazy_transfer_under_mb = skip_lazy_transfer_under_mb,
                **local_expanded_inputs,
            ),
        }
    }
    # print("new_graph", json.dumps(new_graph, indent = 4))

    replaced_output_ids = tuple([json_node_id_str, i] for i in range(_NUM_OUTPUTS))
    return {
        "result": replaced_output_ids,
        "expand": new_graph,
    }


def serialize_obj(serialization: str, obj):
    if serialization in ("torch_pt", "unsafe_torch_pt", "safe_torch_pt"):
        buf = BytesIO()
        torch.save(obj, f = buf)
        buf = buf.getvalue()
    elif serialization == "fancy_safetensors":
        buf = fst_torch.save_bytes(obj, transform_tensor_fn = lambda t: t.detach().clone().contiguous())
    else:
        raise ValueError("Unknown serialization method", serialization)
    return buf


def deserialize_response(serialization: str, result: dict | str | bytes, compression: str | None = None):
    if isinstance(result, dict):
        result_str = result["results"][0]
        data = base64.b64decode(result_str)
    elif isinstance(result, str):
        data = base64.b64decode(result)
    elif isinstance(result, bytes):
        data = result
    else:
        raise ValueError("Invalid result type", type(result), result)

    if compression == "gzip":
        data = gzip.decompress(data)
    elif compression is not None:
        raise ValueError("Unknown compression", compression)

    return deserialize_data(serialization, data)


def deserialize_data(serialization: str, data: bytes):
    if serialization == "unsafe_torch_pt":
        return torch.load(BytesIO(data))
    elif serialization == "safe_torch_pt":
        return torch_safe_load_dict(ZipFile(BytesIO(data), mode = "r"))
    elif serialization == "fancy_safetensors":
        return fst_torch.load(data)
    else:
        raise ValueError("Unknown serialization method", serialization)


def update_toggles_inplace(prompt: dict, toggle: bool, reconnect: bool, label: str, raise_on_existing: bool, input_node_is_local: bool | None = None):
    to_switch = {
        RemoteRunTogglerNode.TYPE_NAME: "inputs_when_off",
        RemoteRunInputNode.TYPE_NAME:   "inputs_when_local",
    }
    reconnected_cnt = disconnected_cnt = 0
    for node_id, node in prompt.items():
        class_type = node["class_type"]
        input_name = to_switch.get(class_type)
        if not input_name:
            continue

        inputs = node.get("inputs")
        if not inputs:
            continue

        if class_type == RemoteRunTogglerNode.TYPE_NAME:
            enabled = inputs.get("enabled")
            if enabled not in TOGGLE_CHOICES:
                raise ValueError("Invalid enabled value", node_id, enabled, TOGGLE_CHOICES)
            if toggle:
                enabled = inputs["enabled"] = TOGGLE_CHOICES[1] if enabled == TOGGLE_CHOICES[0] else TOGGLE_CHOICES[0]

            enabled = enabled == "only_locally"
        else:
            enabled = input_node_is_local

        if (
                reconnect
                and enabled is True
                and inputs.get(input_name) == "disconnected"
                and (disconnected_inputs := inputs.get("_ignore_"))
        ):
            # restore original disconnected inputs for remote side
            try:
                disconnected_inputs = json.loads(disconnected_inputs)
            except Exception as e:
                logger.error("%s RemoteRunToggler.run invalid disconnected_inputs %r: %s", label, disconnected_inputs, e)
                raise ValueError("Invalid disconnected_inputs", node_id, disconnected_inputs, e)

            if disconnected_inputs:
                for inp_name, inp_val in disconnected_inputs.items():
                    if inp_name in inputs:
                        raise ValueError("RemoteRunToggler.run disconnected_inputs input already set", node_id, inp_name, inp_val)
                    inputs[inp_name] = inp_val
                inputs.pop("_ignore_")
                logger.debug("%s reconnected %s RemoteRunToggler inputs on node %r: %s",
                             label, len(disconnected_inputs), node_id, disconnected_inputs)
                reconnected_cnt += 1

        elif enabled is False and inputs.get(input_name) == "disconnected":
            given_inputs = { k: v for k, v in inputs.items() if k.startswith("input_") }
            for k in given_inputs.keys():
                inputs.pop(k)

            if not given_inputs:
                continue

            cur = inputs.get("_ignore_")
            if cur:
                if raise_on_existing:
                    raise ValueError("remote_run update_toggles_inplace overwriting existing _ignore_", node_id, cur, given_inputs)
                else:
                    logger.warning("%s overwriting existing _ignore_ on node_id=%r, cur=%r, new=%r",
                                   label, node_id, cur, given_inputs)

            inputs["_ignore_"] = json.dumps(given_inputs)
            logger.debug("%s disconnected inputs for node_id=%r, inputs=%s",
                         label, node_id, given_inputs)
            disconnected_cnt += 1

    return reconnected_cnt, disconnected_cnt


def retry_dec(total_tries: int = 1, log_exc = True, sleep = None, stop_on = None):
    if isinstance(stop_on, type):
        stop_on = (stop_on,)
    stop_on = stop_on or []

    def make_dec(func):
        def wrapper(*args, **kwargs):
            for i in range(total_tries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if log_exc:
                        logger.exception("function error: fn:%r, retrying %d", func, i)

                    if i == total_tries - 1 or (stop_on and type(e) in stop_on):
                        raise e

                    if sleep:
                        time.sleep(sleep)

        return wrapper

    return make_dec


def _make_adjusted_timeout(total_timeout: float | None, allow_empty: bool = True):
    if not total_timeout:
        if not allow_empty:
            raise ValueError("total_timeout must be > 0")

        def get(wanted_timeout: float, raise_on_timeout = False):
            if raise_on_timeout and wanted_timeout <= 0:
                raise TimeoutError("wanted timeout reached", wanted_timeout)
            return wanted_timeout

        return get

    if total_timeout < 0:
        raise ValueError("total_timeout must be >= 0", total_timeout)

    # ensure current timeout wouldn't exceed total_timeout
    start_ts = time.monotonic()

    def get(wanted_timeout: float, raise_on_timeout = False):
        left_secs = total_timeout - (time.monotonic() - start_ts)
        if raise_on_timeout and left_secs <= 0:
            raise TimeoutError("total timeout reached", total_timeout)
        return min(left_secs, wanted_timeout)

    return get


def remote_execute_prompt(
        base_url: str, prompt: dict,
        own_id: str, extra_output_ids: list[str] | None,
        ws_compression: bool, forward_progress_messages: str,
        binary_response: bool, ws_max_size: int | None, post_max_size: int | None,
        total_timeout: float, short_timeout: float,
        lazy_data: dict | None = None,
        total_tries = 3,
):
    ws_max_size = ws_max_size or 64 * 1024 * 1024

    total_timeout = total_timeout  # can be 0 to disable
    short_timeout = short_timeout or 5  # expected to always be set, so default to low ones in case of unset

    base_url = base_url.rstrip("/")
    base_ws_url = base_url.replace("http://", "ws://")

    client_id = str(uuid4())
    post_nonce = str(uuid4())

    adjusted_timeout_fn = _make_adjusted_timeout(total_timeout or None, allow_empty = True)

    @retry_dec(total_tries, stop_on = TimeoutError)
    def post(timeout):
        post_url = base_url + "/api/prompt"
        apicall = dict(
            client_id = client_id,
            prompt = prompt,
            nonce = post_nonce,  # for safe retry, assumes nonce duplicate node/extension installed
            partial_execution_targets = list(extra_output_ids or []) + [own_id],
        )

        DEBUG_PROMPT_SAVE("post_prompt", prompt) if IS_DEV and IS_DEV_PROMPTSAVE else None

        body = json.dumps(apicall)
        if isinstance(body, str):
            body = body.encode()
        if post_max_size and len(body) > post_max_size:
            raise ValueError(f"Post body too large: {len(body)} bytes, post_max_size={post_max_size}")

        logger.info("Posting prompt to url=%r, size=%d, timeout=%.1f", post_url, len(body), timeout)

        headers = { "Content-Type": "application/json" }
        resp = requests.post(post_url, data = body, headers = headers, timeout = adjusted_timeout_fn(timeout, True))
        if resp.status_code != 200:
            raise ValueError("Failed to post prompt", post_url, resp.status_code, resp.content, apicall)

        obj = resp.json()
        return obj["prompt_id"]

    @retry_dec(total_tries, stop_on = (TimeoutError, ValueError))
    def post_lazy_data(key, status, body: bytes | None, timeout):
        params = dict(
            key = key,
            status = status,
        )
        post_url = base_url + "/api/remote_run/data/"
        headers = { }
        logger.info("Posting requested lazy data to url=%r, params=%r, size=%d, timeout=%.1f",
                    post_url, params, len(body or ""), timeout)
        resp = requests.post(post_url, params = params, data = body, headers = headers, timeout = adjusted_timeout_fn(timeout, True))
        if resp.status_code != 200:
            raise ValueError("Failed to post prompt", post_url, resp.status_code, resp.content)
        return resp.content

    ws_url = base_ws_url + f"/ws?clientId={client_id}"

    websocket = None
    # create websocket connection first, then send prompt POST and wait for result/error/timeout on ws
    try:
        @retry_dec(total_tries, stop_on = TimeoutError)
        def connect():
            return websockets.sync.client.connect(
                ws_url,
                open_timeout = adjusted_timeout_fn(short_timeout), close_timeout = 5,
                max_size = ws_max_size,
                compression = "deflate" if ws_compression else None,
            )

        logger.debug("Connecting to websocket url=%r", ws_url)
        websocket = connect()
        prompt_id = post(short_timeout)

        forward_ws_messages = forward_progress_messages == "on"
        forwarded_types = { "executing", "progress_state" }

        while True:
            message = websocket.recv(timeout = adjusted_timeout_fn(total_timeout, True))
            if isinstance(message, bytes):
                if not binary_response:
                    logger.warning("Ignoring binary websocket message, expected text: size=%d", len(message))
                else:
                    if len(message) < 4:
                        logger.warning("Ignoring too short binary websocket message, expected longer: size=%d", len(message))
                        continue
                    message_type = int.from_bytes(message[:4], "big")
                    if message_type == BinaryResponseMessageID.BINARY_RESPONSE_MESSAGE_ID.value:
                        logger.info("Got binary response websocket message: size=%d", len(message))
                        rest = message[4:]
                        return rest
                    elif message_type == BinaryResponseMessageID.GZIPPED_BINARY_RESPONSE_MESSAGE_ID.value:
                        logger.info("Got gzipped binary response websocket message: size=%d", len(message))
                        comp = message[4:]
                        gzrest = gzip.decompress(comp)
                        return gzrest
                    else:
                        logger.warning("Ignoring unknown binary websocket message type: type=%d, size=%d", message_type, len(message))
                        continue
                continue

            message_len = len(message or "")
            message_obj = json.loads(message)
            # print("ws message", message_obj) if IS_DEV else None

            msg_type = message_obj.get("type")
            if msg_type == "NEED_DATA":
                logger.info("Got NEED_DATA message for: %s", message_obj)
                key = None
                try:
                    req_obj = message_obj["data"]
                    key = req_obj["key"]
                    if lazy_data and key in lazy_data:
                        data = lazy_data.pop(key)
                        post_lazy_data(key, "data", data, short_timeout)
                        continue
                    else:
                        logger.error("No lazy data found for key %r, sending empty response", key)
                        post_lazy_data(key, "not_found", None, short_timeout)
                except Exception:
                    logger.exception("Invalid NEED_DATA message data")
                    if key:
                        post_lazy_data(key, "error", None, short_timeout)

                raise ValueError("Invalid NEED_DATA message", message_obj)

            own_prompt_id = message_obj.get("data", { }).get("prompt_id") == prompt_id
            if own_prompt_id:
                if (
                        forward_ws_messages
                        and msg_type in forwarded_types
                        and (data := message_obj.get("data"))
                ):
                    node = data.get("node")
                    ignore = msg_type == "executing" and node == own_id
                    if not ignore:
                        try:
                            import server
                            instance = server.PromptServer.instance
                            client_id = instance.client_id
                            logger.debug("Forwarding remote prompt server message type=%r, node=%r, client_id=%r, size=%d",
                                         msg_type, node, client_id, message_len)
                            instance.send_sync(msg_type, data, client_id)
                        except Exception:
                            logger.exception("Failed to forward remote prompt server message type=%r, node=%r, client_id=%r",
                                             msg_type, node, client_id)

                if msg_type == "executed":
                    logger.debug("got executed message for own prompt_id=%r, size=%d", prompt_id, message_len)
                    data = message_obj["data"]
                    if data["node"] == own_id:
                        result = data["output"]
                        return result
                elif own_prompt_id and msg_type == "execution_error":
                    raise ValueError("remote comfy prompt execution error", message_obj, base_url, prompt)
    except TimeoutError:
        # TODO: interrupt execution of that prompt in case of timeout?? always or option
        # also not sure if specific prompt interrupt is easily possible
        raise

    finally:
        if websocket is not None:
            threading.Thread(target = websocket.close, daemon = True).start()


class RemoteRunSetNumOutputsNode():
    DISPLAY_NAME = "RemRun Set Num Outputs"

    CATEGORY = "Remote Run"
    FUNCTION = "run"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "num": ("INT", { "default": 10, "min": 1, "max": 100, "step": 1 }),
            },
        }

    RETURN_TYPES = ()

    def run(self, num: int):
        global _NUM_OUTPUTS
        logger.info("Changing RemoteRun number of inputs/outputs from %d to %d", _NUM_OUTPUTS, num)
        _NUM_OUTPUTS = num
        return { }


def filter_definitions(definitions: dict, kwargs):
    type_regexes = [v for k, v in kwargs.items() if k.startswith("class_type_include_regex_") and v]
    display_name_regexes = [v for k, v in kwargs.items() if k.startswith("display_name_include_regex_") and v]
    ignore_type_regexes = [v for k, v in kwargs.items() if k.startswith("class_type_ignore_regex_") and v]
    ignore_display_name_regexes = [v for k, v in kwargs.items() if k.startswith("display_name_ignore_regex_") and v]
    category_name_regexes = [v for k, v in kwargs.items() if k.startswith("category_name_regex_") and v]
    category_name_ignore_regexes = [v for k, v in kwargs.items() if k.startswith("category_name_ignore_regex_") and v]

    type_regexes = [re.compile(r, flags = re.IGNORECASE) for r in type_regexes]
    display_name_regexes = [re.compile(r, flags = re.IGNORECASE) for r in display_name_regexes]
    ignore_type_regexes = [re.compile(r, flags = re.IGNORECASE) for r in ignore_type_regexes]
    ignore_display_name_regexes = [re.compile(r, flags = re.IGNORECASE) for r in ignore_display_name_regexes]
    category_name_regexes = [re.compile(r, flags = re.IGNORECASE) for r in category_name_regexes]
    category_name_ignore_regexes = [re.compile(r, flags = re.IGNORECASE) for r in category_name_ignore_regexes]

    matching = { }
    for class_type, node_class in definitions.items():
        if type_regexes and not any(r.search(class_type) for r in type_regexes):
            continue
        if ignore_type_regexes and any(r.search(class_type) for r in ignore_type_regexes):
            continue

        display_name = node_class.get("display_name", None)
        if display_name and display_name_regexes and not any(r.search(display_name) for r in display_name_regexes):
            continue
        if display_name and ignore_display_name_regexes and any(r.search(display_name) for r in ignore_display_name_regexes):
            continue

        category = node_class.get("category", None)
        if category and category_name_regexes and not any(r.search(category) for r in category_name_regexes):
            continue
        if category and category_name_ignore_regexes and any(r.search(category) for r in category_name_ignore_regexes):
            continue

        matching[class_type] = node_class
    return matching


class RemoteRunAddRemoteNodeDefinitionsNode():
    DISPLAY_NAME = "RemRun Add Remote Node Definitions"

    CATEGORY = "Remote Run"
    FUNCTION = "run"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "remote_url":           ("STRING", {
                    "default": "http://127.0.0.1:8189/",
                    "tooltip": "ComfyUI instance URL",
                }),
                "request_timeout":      ("FLOAT", {
                    "tooltip": "Request timeout in seconds",
                    "default": 15.0, "min": 0.1, "max": 24 * 3600.0, "step": 0.1
                }),
                "dry_run":              ("BOOLEAN", { "default": False }),
                "display_name_prefix":  ("STRING", { "default": "[RemoteNd] " }),
                "display_name_suffix":  ("STRING", { "default": "" }),
                "category_name_prefix": ("STRING", { "default": "[RemoteNd]/" }),
                "category_name_suffix": ("STRING", { "default": "" }),
                "overwrite_existing":   ("BOOLEAN", { "default": False }),
                "is_changed":           ("BOOLEAN", { "default": False }),
            },
            "optional": {
                **{ f"class_type_include_regex_{i}": ("STRING", { }) for i in range(5) },
                **{ f"display_name_include_regex_{i}": ("STRING", { }) for i in range(5) },
                **{ f"class_type_ignore_regex_{i}": ("STRING", { }) for i in range(5) },
                **{ f"display_name_ignore_regex_{i}": ("STRING", { }) for i in range(5) },
                **{ f"category_name_regex_{i}": ("STRING", { }) for i in range(5) },
                **{ f"category_name_ignore_regex_{i}": ("STRING", { }) for i in range(5) },
            },
        }

    RETURN_TYPES = ()

    @classmethod
    def IS_CHANGED(cls, is_changed: bool = False, **_kwargs):
        return time.time_ns() if is_changed else None

    def run(self, remote_url: str, request_timeout: float, is_changed: float | None = None, **kwargs):
        if not remote_url.endswith("/"):
            remote_url += "/"
        if not remote_url.startswith("http"):
            raise ValueError("Invalid remote_url, must start with http:// or https://", remote_url)

        url = remote_url + "object_info"
        resp = requests.get(url, timeout = request_timeout)
        if not resp.status_code == 200:
            raise ValueError("Failed to get remote object info", url, resp.status_code, resp.content)
        obj = resp.json()
        logger.info("Got %s node definitions from remote %r", len(obj or { }), remote_url)
        self.add_definitions(obj, kwargs)
        return { }

    def add_definitions(self, definitions: dict, kwargs):
        matching = filter_definitions(definitions, kwargs)
        logger.info("Found %d matching node definitions of %s", len(matching), len(definitions))
        if not matching:
            return

        stats = collections.defaultdict(int)
        for class_type, node_obj in matching.items():
            try:
                self.add_definition(class_type, node_obj, stats, **kwargs)
            except Exception as e:
                logger.error(f"Error adding node definition for class {class_type!r}: {e}")
                stats["errored"] += 1

        added, overwrote, skipped, errored = stats["added"], stats["overwrote"], stats["skipped"], stats["errored"]
        dry_run_label = "DRYRUN: " if kwargs.get("dry_run") else ""
        logger.info("%s", f"{dry_run_label}Processed {len(matching)} node definitions "
                          f"(added: {added}, overwrote: {overwrote}, skipped: {skipped}, errors: {errored}) of {len(definitions)} total definitions")

    def add_definition(self, class_type: str, defn: dict, stats: dict,
                       dry_run: bool = False,
                       overwrite_existing: bool = False,
                       display_name_prefix: str = "",
                       display_name_suffix: str = "",
                       category_name_prefix: str = "",
                       category_name_suffix: str = "",
                       **_kwargs,
                       ):
        dry_run_label = "DRYRUN: " if dry_run else ""
        stats = stats if stats is not None else { }
        existed = class_type in nodes.NODE_CLASS_MAPPINGS
        if existed and not overwrite_existing:
            logger.debug("%s", f"{dry_run_label}Skipped existing node class: {class_type}")
            stats["skipped"] += 1
            return

        display_name = defn.get("display_name", class_type)
        if display_name_prefix:
            display_name = f"{display_name_prefix}{display_name}"
        if display_name_suffix:
            display_name = f"{display_name}{display_name_suffix}"

        category = defn.get("category") or ""
        if category_name_prefix:
            category = f"{category_name_prefix}{category}"
        if category_name_suffix:
            category = f"{category}{category_name_suffix}"

        def tup_get(key):
            val = defn.get(key)
            if val is None:
                return None
            if isinstance(val, list):
                return tuple(val)
            raise ValueError(f"Invalid {key} value, must be list or None", val)

        input_types = defn.get("input")

        @classmethod
        def INPUT_TYPES(cls):
            return input_types

        new_class = type(class_type, (object,), { })
        setattr(new_class, "INPUT_TYPES", INPUT_TYPES)

        for name, val in (
                ("DISPLAY_NAME", display_name),
                ("CATEGORY", category),
                ("RETURN_TYPES", tup_get("output")),
                ("RETURN_NAMES", tup_get("output_name")),
                ("DESCRIPTION", defn.get("description")),
                ("OUTPUT_NODE", defn.get("output_node")),
        ):
            if val is not None:
                setattr(new_class, name, val)

        if not dry_run:
            nodes.NODE_CLASS_MAPPINGS[class_type] = new_class
            if display_name is not None:
                nodes.NODE_DISPLAY_NAME_MAPPINGS[class_type] = display_name

        stat_name = "overwrote" if existed else "added"
        logger.info("%s", f"{dry_run_label}{stat_name.capitalize()} node class: {class_type} ({display_name})")
        stats[stat_name] += 1


def add_server_prompt_nonce_handler():
    logger.info("Adding promptserver remote_run nonce no duplicate prompt handler")
    import server

    nonce_ordered = collections.deque()
    nonce_set = set()
    NONCE_MAX = 1000

    def prompt_handler(original_request: dict):
        nonce = original_request.get("nonce")
        if nonce is None:
            return original_request

        if nonce in nonce_set:
            logger.info("Duplicate nonce found, ignoring prompt: %r", nonce)
            return { }

        nonce_set.add(nonce)
        while len(nonce_set) > NONCE_MAX:
            oldest = nonce_ordered.popleft()
            nonce_set.remove(oldest)
        return original_request

    if os.environ.get("HOTRELOAD"):
        to_del = [i for i in server.PromptServer.instance.on_prompt_handlers if getattr(i, "rr_nonce_nodup_clearable", False)]
        for i in to_del:
            print("hotreload clearing old remote run nonce prompt handler", i)
            server.PromptServer.instance.on_prompt_handlers.remove(i)
        prompt_handler.rr_nonce_nodup_clearable = True

    server.PromptServer.instance.add_on_prompt_handler(prompt_handler)


def add_server_prompt_input_disconnect_handler():
    logger.info("Adding promptserver remote_run input disconnect handler")
    import server

    def prompt_handler(original_request: dict):
        if not isinstance(original_request.get("prompt"), dict):
            return original_request

        original_request = copy.deepcopy(original_request)
        prompt = original_request["prompt"]
        DEBUG_PROMPT_SAVE("rr_prompt_handler__pre_rewrite_prompt__", prompt) if IS_DEV and IS_DEV_PROMPTSAVE else None

        rcon, dscon = update_toggles_inplace(prompt, False, False, "input_disconnect_handler",
                                             raise_on_existing = True, input_node_is_local = False)  # False so it gets disconnected
        if rcon or dscon:
            logger.info("remote_run input disconnect handler, reconnected %d, disconnected %d nodes", rcon, dscon)

        DEBUG_PROMPT_SAVE("rr_prompt_handler__rerewritten_prompt", prompt) if IS_DEV and IS_DEV_PROMPTSAVE else None
        return original_request

    if os.environ.get("HOTRELOAD"):
        to_del = [i for i in server.PromptServer.instance.on_prompt_handlers if getattr(i, "rr_input_discon_clearable", False)]
        for i in to_del:
            print("hotreload clearing old remote run input disconnect prompt handler", i)
            server.PromptServer.instance.on_prompt_handlers.remove(i)
        prompt_handler.rr_input_discon_clearable = True
        server.PromptServer.instance.on_prompt_handlers.insert(0, prompt_handler)
    else:
        server.PromptServer.instance.add_on_prompt_handler(prompt_handler)


if os.environ.get("ADD_REMOTE_RUN_NONCE_HANDLER") == "1":
    logger.info("ADD_REMOTE_RUN_NONCE_HANDLER=1, adding remote_run no duplicate nonce handler!")
    try:
        add_server_prompt_nonce_handler()
    except Exception as e:
        logger.exception("Failed to add remote_run no duplicate nonce handler")
else:
    logger.info("skipping remote run_no duplicate nonce handler, ADD_REMOTE_RUN_NONCE_HANDLER=1 to enable.")

if os.environ.get("SKIP_REMOTE_RUN_INPUT_DISCONNECT_HANDLER") != "1":
    logger.info("SKIP_REMOTE_RUN_INPUT_DISCONNECT_HANDLER=1 not given, adding remote_run input disconnect handler!")
    try:
        add_server_prompt_input_disconnect_handler()
    except Exception as e:
        logger.exception("Failed to add remote_run input disconnect handler")
else:
    logger.info("skipping remote_run input disconnect handler due to env var.")

if os.environ.get("SKIP_REMOTE_RUN_ADD_DATA_ROUTE") != "1":
    logger.info("SKIP_REMOTE_RUN_ADD_DATA_ROUTE=1 not given, adding remote_run data route handler!")
    try:
        setup_remote_run_api_route(expected_set_up = False)
    except Exception as e:
        logger.exception("Failed to add remote_run data route handler")
else:
    logger.info("skipping remote_run data route handler due to env var.")


def node_mappings(classes):
    make_type_name = lambda cn: f"RAT_{cn.removesuffix('Node')}"

    class_map, name_map = { }, { }
    for i in classes:
        type_name = getattr(i, "TYPE_NAME", make_type_name(i.__name__))
        class_map[type_name] = i
        name_map[type_name] = getattr(i, "DISPLAY_NAME", type_name)

    return class_map, name_map


REMOTE_RUN_RUN_NODES = [
    RemoteRunInputNode,
    RemoteRunInputOutputNode,
    RemoteRunJsonNode,
]
NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = node_mappings(
    (
        RemoteRunStartNode,
        *REMOTE_RUN_RUN_NODES,
        RemoteRunTogglerNode,
        RemoteRunSerializerOutNode, RemoteRunDeserializerOutNode,

        RemoteRunSetNumOutputsNode,
        RemoteRunAddRemoteNodeDefinitionsNode,
    )
)
