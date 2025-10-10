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

import base64
import collections
import copy
import enum
import functools
import gzip
import math
import threading
import time
import os
from io import BytesIO

from uuid import uuid4
from zipfile import ZipFile

import execution
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

jdumps = lambda data: json.dumps(data).replace("\n", "")

_NUM_OUTPUTS = int(os.environ.get("RAT_REMOTE_RUN_NUM_OUTPUTS") or 5)
logger.info("%s", f"RemoteRun nodes using _NUM_OUTPUTS={_NUM_OUTPUTS}")

TOGGLE_CHOICES = ("only_locally", "only_remotely")
INPUTS_OFF_OPTIONS = ["lazy", "disconnected"]
BLOCK_OPTIONS = ["error", "block_execution_silent", "block_execution_verbose", "passthrough", "return_none"]


class BinaryResponseMessageID(enum.IntEnum):
    BINARY_RESPONSE_MESSAGE_ID = 12341119
    GZIPPED_BINARY_RESPONSE_MESSAGE_ID = 12341120


def serialization_load_input(name = "serialization"):
    return { name: (["safe_torch_pt", "unsafe_torch_pt", "fancy_safetensors"], { "default": "fancy_safetensors" }) }


def response_input(name = "response"):
    return { name: (["base64_result", "binary"], { "default": "base64_result" }) }


def ws_settings_inputs():
    return {
        "ws_compression":            ("BOOLEAN", { "default": False }),
        # strings to make it easier to add more in case of wanting a partial option without breaking backwards compatibility
        "forward_progress_messages": (["on", "off"], { "default": "on" }),
    }


def shared_input_settings():
    return {
        "max_size_mb":      ("INT", { "default": 32, "min": 0, "max": 8 * 1024, "step": 1 }),
        "gzip_compression": ("BOOLEAN", { "default": False }),
        "gzip_level":       ("INT", { "default": 9, "min": 1, "max": 9, "step": 1 }),
    }


class RemoteRunSerializerOutNode():
    TYPE_NAME = "RAT_RemoteRunSerializerOut"
    DISPLAY_NAME = "Remote Run Serializer Output (Internal)"

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
                **shared_input_settings(),
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


class RemoteRunDeserializerOutNode():
    TYPE_NAME = "RAT_RemoteRunDeserializerOut"
    DISPLAY_NAME = "Remote Run Deserializer Output (Internal)"

    CATEGORY = "Remote Run/__Internal__/"
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                **serialization_load_input(),
                "data": ("STRING", { }),
            },
            "optional": {
                "is_changed": ("BOOLEAN", { "default": False }),
            },
        }
        return inputs

    RETURN_TYPES = tuple([CoIO.ANY] * _NUM_OUTPUTS)

    def run(self, serialization: str, data: str, **_kwargs):
        obj = deserialize_response(serialization, data)
        print("RemoteRunDeserializerOutNode", obj)
        return tuple(obj.get(i) for i in range(_NUM_OUTPUTS))


class RemoteRunToggler():
    """
    This is a Input Toggle Switch that can be used to enable/disable certain parts of a graph
    in the exact opposite way between the local and remote side using lazy evaluation or prompt preprocessing.

    If the switch run_side is set to enabled it will run all inputs, otherwise set them all to off/lazy
    or disconnected before prompt execution, depending on [inputs_when_off] setting.
    When a RemoteRun runner node encounters one of these switches in a prompt it just toggles
    the run_side value so the remote side will do the exact opposite of the local one,
    without even knowing it's the remote side.
    """
    TYPE_NAME = "RAT_RemoteRunToggler"
    DISPLAY_NAME = "Remote Run Toggle"

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
            "tooltip": "timeout in seconds",
            "default": 90.0, "min": 0.1, "max": 24 * 3600.0, "step": 0.1
        }),
        "request_timeout": ("FLOAT", {
            "tooltip": "short timeout in seconds",
            "default": 10.0, "min": 0.1, "max": 24 * 3600.0, "step": 0.1
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
    DISPLAY_NAME = "Remote Run Input Graph(s)"

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
                **shared_input_settings(),
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
            dynprompt: DynamicPrompt = None, own_id = None, _ignore_ = None,
            **_kwargs,
            ):
        remote_prompt = { i: dynprompt.get_node(i) for i in dynprompt.all_node_ids() }
        return partial_json_expansion(
            remote_prompt, own_id, remote_url, serialization,
            total_timeout, request_timeout,
            max_size_mb, gzip_compression, gzip_level,
            ws_compression, forward_progress_messages, is_changed, response,
            run_outputs_connected_to_inputs,
        )


class RemoteRunInputOutputNode(RemoteRunInputNode):
    TYPE_NAME = "RAT_RemoteRunInputOutput"
    DISPLAY_NAME = "Remote Run Input Graph(s) Output"
    OUTPUT_NODE = True


class RemoteRunJsonNode():
    TYPE_NAME = "RAT_RemoteRunJson"
    DISPLAY_NAME = "Remote Run JSON"

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
                "max_size_mb": shared_input_settings()["max_size_mb"],
                **response_input(),
                **ws_settings_inputs(),
                "is_changed":  ("BOOLEAN", { "default": False }),
            },
        }
        return inputs

    RETURN_TYPES = tuple([CoIO.ANY] * _NUM_OUTPUTS)

    def run(self,
            remote_url: str, total_timeout: float, request_timeout: float, JSON: str, serialization: str,
            max_size_mb: int | None = None, response = None,
            ws_compression: bool = False, forward_progress_messages: str = "off",
            **kwargs,
            ):
        max_size_mb = max_size_mb or None
        run_obj = json.loads(JSON)
        extra_output_ids = run_obj.get("extra_output_ids") or None
        run_prompt = run_obj["prompt"]
        data_total = 0
        for node_id, node in run_prompt.items():
            # serialize needed inputs to this node and set up
            # Deserializer data output nodes with the serialized data in the remote prompt
            if node["class_type"] == RemoteRunDeserializerOutNode.TYPE_NAME:
                config = node.pop("deserializer_config")
                outputs = config["outputs"]

                data = { }
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
                        data[output_num] = kwargs[local_input_name]
                    elif output_data_type == "CONSTANT":
                        data[output_num] = output_data_value
                    else:
                        raise ValueError("Invalid output data", output_data, config)

                data = serialize_obj(serialization, data)
                data = base64.b64encode(data).decode()

                data_total += len(data)
                if max_size_mb and data_total > max_size_mb * 1024 * 1024:
                    raise ValueError(f"Serialized data too large: {data_total} bytes, max_size_mb={max_size_mb}")

                node["inputs"] = {
                    "serialization": serialization,
                    "data":          data,
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
    elif option == "return none":
        return tuple(None for _ in range(num_outputs))
    else:
        raise ValueError(f"Invalid option value: {option!r}")


class RemoteRunStartNode():
    TYPE_NAME = "RAT_RemoteStart"
    DISPLAY_NAME = "Remote Run Start From Here ->"

    CATEGORY = "Remote Run"
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "optional": {
                **dict((f"input_{num}", (CoIO.ANY, { })) for num in range(_NUM_OUTPUTS)),
                "remote_run_dependent_outputs": (["ignore", "run"], { "default": "ignore" }),
                "inputs_when_local":            (INPUTS_OFF_OPTIONS, { "default": "error" }),
                "outputs_when_local":           (BLOCK_OPTIONS, { "default": "error" }),
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


def get_extra_output_ids_dependent_on(prompt: dict, node_ids: list[str]):
    # get all nodes that are directly or indirectly linked to the outputs of the given node_ids
    # (generally all RemoteRunStartNodes), stop going down the chain when hitting another remote_run
    # start or input/json node.

    node_ids = set(node_ids)
    stop_at_types = { i.TYPE_NAME for i in REMOTE_RUN_RUN_NODES + [RemoteRunStartNode] }
    stop_at_ids = [i for i, n in prompt.items() if n["class_type"] in stop_at_types]
    all_stop_at_ids = set(stop_at_ids) | node_ids
    deps = get_full_dependents(prompt, all_stop_at_ids)

    extra_output_ids = set()
    for parent_id, all_deps in deps.items():
        if not all_deps:
            continue
        if parent_id not in node_ids:
            continue
        extra_output_ids.update(all_deps)

    return sorted(extra_output_ids)


def get_full_dependents(prompt: dict, important_parent_node_ids: set[str]) -> dict[str, set[str]]:
    # For each important_parent_node_id get all nodes that are directly or indirectly dependent on it
    # but stop going down the chain when hitting another important_parent_node_id
    important_parent_node_ids = set(important_parent_node_ids)

    inputs_map = { }
    for nid, node in prompt.items():
        inputs = set()
        for v in node["inputs"].values():
            if isinstance(v, list) and len(v) == 2 and v[0] in prompt:
                inputs.add(v[0])
        inputs_map[nid] = inputs

    @functools.cache
    def get_all_parents(nid):
        if nid in important_parent_node_ids:
            return set()

        parents = set()
        for inp in inputs_map.get(nid) or []:
            parents.add(inp)
            parents.update(get_all_parents(inp))
        return parents

    res = { i: set() for i in important_parent_node_ids }
    for nid in prompt.keys():
        all_parents_until_parent_node_id_or_end = get_all_parents(nid)
        important = all_parents_until_parent_node_id_or_end & important_parent_node_ids
        for p in important:
            res[p].add(nid)

    return res


def get_input_graph_nodes(prompt: dict, root_node_id: str, add_dependent_nodes: bool) -> set[str]:
    input_nodes = { root_node_id }

    def _nodes(node_id: str, seen):
        if node_id in seen:
            return
        seen.add(node_id)

        node = prompt[node_id]
        inputs = node["inputs"]
        for inp in inputs.values():
            if isinstance(inp, list):
                input_nodes.add(inp[0])
                _nodes(inp[0], seen)

    _nodes(root_node_id, set())

    if not add_dependent_nodes:
        return input_nodes

    full_deps = get_full_dependents(prompt, input_nodes)
    for parent_id, all_deps in full_deps.items():
        if not all_deps or parent_id == root_node_id:
            continue
        input_nodes.update(all_deps)

    return input_nodes


def partial_json_expansion(
        remote_prompt: dict, root_node_id: str, remote_url: str, serialization: str,
        total_timeout: float, request_timeout: float,
        max_size_mb: int | None, gzip_compression: bool, gzip_level: int,
        ws_compression: bool, forward_progress_messages: str, is_changed: bool, response: str | None,
        run_outputs_connected_to_inputs: str,
):
    """
        When there are RemoteStart nodes then go from:
            A -> B -> C -> RemoteStart -> D -> E -> F -> RemoteRunInput -> G -> H -> I
            to
            A -> B -> C -> RemoteRunJson(run_remotely="D -> E -> F") -> G -> H -> I

            Build an expanded graph to replace the RemoteRunInput node with a RemoteRunJson node.
            Copy the whole input graph into RemoteRunInput node until a RemoteStart node is hit,
            and replace that whole chunk with a RemoteRunJson node that will run that part instead.
            The RemoteRunJson gets linked to all the inputs of the RemoteStart nodes so it can serialize them and send them
            to the remote instance as part of the remote prompt.
        If no RemoteStart nodes take all the nodes that the RemoteRunInput node depends on and
        optionally also all nodes that depend on the RemoteRunInput node if run_outputs_connected_to_inputs is "run".
        So from:
                A -> B -> C -> D -> RemoteRunInput -> G -> H -> I
                A -> SaveImage
            to:
                RemoteRunJson(run_remotely="A -> B -> C -> D") -> G -> H -> I
            or with run_outputs_connected_to_inputs = "run" to:
                RemoteRunJson(run_remotely="A -> SaveImage; A -> B -> C -> D", run_extra_output=["D"]) -> G -> H -> I

    """
    remote_prompt = copy.deepcopy(remote_prompt)
    rcon, dscon = update_toggles_inplace(remote_prompt, True, True, "remote_run_input_toggle", False)

    next_node_id = make_counter(_max_id(remote_prompt) + 1, str)

    if rcon or dscon:
        logger.info("%s", f"partial_json_expansion {root_node_id=} toggled {rcon + dscon} nodes, "
                          f"reconnected {rcon},  disconnected {dscon}")

    extra_output_ids = None
    local_expanded_inputs = { }

    start_node_ids, middle_node_ids = input_start_nodes(remote_prompt, root_node_id)
    if start_node_ids:
        # The D -> E -> F part(s), anything before the RemoteRunInput node up until to any RemoteRunStart nodes
        # (plus anything hitting the top).
        small_prompt = { i: remote_prompt[i] for i in (middle_node_ids | start_node_ids) }

        # For the remote prompt: just replace each RemoteStart node with a Deserializer node that outputs
        # the serialized and sent over inputs.
        # On the local side as there can be multiple RemoteStart nodes all the inputs to the RemoteStart nodes have to also
        # be sent to the newly expanded RemoteRunJson node and kept track of which goes to which Deserializer node output.

        next_json_input_num = make_counter(0)

        remote_run_dependent_outputs_of = []
        # On remote prompt change all RemoteRunStart nodes to Deserializer nodes
        for start_id in start_node_ids:
            # TODO: ignore inputs for outputs that aren't used, currently serialized/sent for nothing

            start_node = small_prompt[start_id]
            start_inputs = start_node["inputs"]
            node_remote_run_dependent_outputs = start_inputs.get("remote_run_dependent_outputs")
            if (node_remote_run_dependent_outputs or "").lower() == "run":
                remote_run_dependent_outputs_of.append(start_id)

            des = { }
            for input_name, input_value in start_inputs.items():
                if not input_name.startswith("input_"):
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
                    "outputs": des,
                },
            }

        if remote_run_dependent_outputs_of:
            # Add all nodes that are between a StartNode and either free floating like Output nodes or until hitting a RemoteRunInput node.
            # This is all dependent nodes not just Output ones but no need to check for only output ones since any normal non output
            # nodes given will just be ignored by the prompt handling (currently).

            extra_output_ids = get_extra_output_ids_dependent_on(remote_prompt, remote_run_dependent_outputs_of)
            logger.info("partial_json_expansion %s: remote_run_dependent_outputs_of=%s -> extra_output_ids=%s",
                        root_node_id, remote_run_dependent_outputs_of, (len(extra_output_ids), extra_output_ids))

            for eid in extra_output_ids:
                if eid in remote_prompt:
                    continue
                small_prompt[eid] = copy.deepcopy(remote_prompt[eid])

        remote_prompt = small_prompt
    else:
        add_outputs_of_inputs = (run_outputs_connected_to_inputs or "").lower() == "run"
        input_graph_ids = get_input_graph_nodes(remote_prompt, root_node_id, add_dependent_nodes = add_outputs_of_inputs)
        extra_output_ids = input_graph_ids if add_outputs_of_inputs else None
        remote_prompt = { i: remote_prompt[i] for i in input_graph_ids }

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

    remote_obj = dict(
        prompt = remote_prompt,
        extra_output_ids = list(extra_output_ids) or None,
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


def input_start_nodes(prompt: dict, start_id: str):
    start_nodes = set()
    middle_node_ids = set()

    def _nodes(node_id: str):
        if node_id in middle_node_ids:
            return
        middle_node_ids.add(node_id)

        node = prompt[node_id]
        if node["class_type"] == RemoteRunStartNode.TYPE_NAME:
            start_nodes.add(node_id)
            return

        inputs = node["inputs"]
        for inp in inputs.values():
            if isinstance(inp, list):
                _nodes(inp[0])

    _nodes(start_id)

    middle_node_ids = middle_node_ids - start_nodes
    return start_nodes, middle_node_ids


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


def deserialize_response(serialization: str, result: dict | str | bytes):
    if isinstance(result, dict):
        result_str = result["results"][0]
        data = base64.b64decode(result_str)
    elif isinstance(result, str):
        data = base64.b64decode(result)
    else:
        data = result
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


def output_nodes_transform(prompt: dict, output_nodes: str, keep_nodes: set[str] = None) -> dict:
    if output_nodes == "Remove":
        output_ids = set(prompt_outputs(prompt)) - (keep_nodes or set())
        return { k: v for k, v in prompt.items() if k not in output_ids }

    return prompt


def build_full_prompt(
        remote_prompt: dict, start_id: str, response: str | None,
        max_size_mb: int | None, gzip_compression: bool, gzip_level: int,
        is_changed: bool,

) -> dict:
    own_node = remote_prompt[start_id]

    # simple = True
    simple = False
    if simple:
        # Simple method: Copy full current prompt and just delete all outputs from it,
        # then replace own node with an input serializing & returning node.
        # Only the input nodes to this should be run in theory unless any extension/node does anything weird.

        run_prompt = copy.deepcopy(remote_prompt)
        output_ids = prompt_outputs(run_prompt)
        run_prompt = { k: v for k, v in run_prompt.items() if k not in output_ids }
    else:
        # Go though input node graphs and make new prompt only from nodes that lead to the inputs of this node.
        input_nodes = set()

        def _nodes(node_id: str, seen):
            if node_id in seen:
                return
            seen.add(node_id)

            node = remote_prompt[node_id]
            inputs = node["inputs"]
            for inp in inputs.values():
                if isinstance(inp, list):
                    input_nodes.add(inp[0])
                    _nodes(inp[0], seen)

        _nodes(start_id, set())
        run_prompt = { i: remote_prompt[i] for i in input_nodes }

    inputs = copy.deepcopy(own_node["inputs"])
    inputs = { k: v for k, v in inputs.items() if k.startswith("input_") or k == "serialization" }
    inputs.update(dict(
        max_size_mb = max_size_mb,
        response = response,
        gzip_compression = gzip_compression,
        gzip_level = gzip_level,
        is_changed = is_changed,
    ))

    # replace own node with input serializer output node
    run_prompt[start_id] = {
        "inputs":     inputs,
        "class_type": RemoteRunSerializerOutNode.TYPE_NAME,
    }

    return run_prompt


def update_toggles_inplace(prompt: dict, toggle: bool, reconnect: bool, label: str, raise_on_existing: bool, input_node_is_local: bool = True):
    to_switch = {
        RemoteRunToggler.TYPE_NAME:   "inputs_when_off",
        RemoteRunInputNode.TYPE_NAME: "inputs_when_local",
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

        if class_type == RemoteRunToggler.TYPE_NAME:
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
                and enabled
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

        elif not enabled and inputs.get(input_name) == "disconnected":
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


def prompt_outputs(prompt: dict) -> list[str]:
    valid = execution.validate_prompt(prompt)
    if not valid[0]:
        if isinstance(valid[1], dict) and valid[1].get("type") == "prompt_no_outputs":
            return []
        raise Exception(f"Prompt validation failed, couldn't get output node ids: {valid}, {prompt}")
    outputs = valid[2]
    return outputs


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


def remote_execute_prompt(
        base_url: str, prompt: dict,
        own_id: str, extra_output_ids: list[str] | None,
        ws_compression: bool, forward_progress_messages: str,
        binary_response: bool, ws_max_size: int | None, post_max_size: int | None,
        total_timeout: float, short_timeout: float,
        total_tries = 3,
):
    ws_max_size = ws_max_size or 64 * 1024 * 1024

    total_timeout = total_timeout or 10  # expected to always be set, so default to low ones in case of unset
    short_timeout = short_timeout or 5

    base_url = base_url.rstrip("/")
    base_ws_url = base_url.replace("http://", "ws://")

    client_id = str(uuid4())
    post_nonce = str(uuid4())

    def _make_adjusted_timeout(total_timeout: float):
        # ensure current timeout wouldn't exceed total_timeout
        start_ts = time.monotonic()

        def get(wanted_timeout: float, raise_on_timeout = False):
            left_secs = total_timeout - (time.monotonic() - start_ts)
            if raise_on_timeout and left_secs <= 0:
                raise TimeoutError("total timeout reached", total_timeout)
            return min(left_secs, wanted_timeout)

        return get

    adjusted_timeout_fn = _make_adjusted_timeout(total_timeout)

    @retry_dec(total_tries, stop_on = TimeoutError)
    def post(timeout):
        post_url = base_url + "/api/prompt"
        apicall = dict(
            client_id = client_id,
            prompt = prompt,
            nonce = post_nonce,  # for safe retry, assumes nonce duplicate node/extension installed
            partial_execution_targets = list(extra_output_ids or []) + [own_id],
        )

        if os.environ.get("DEV") == "1":
            from pathlib import Path
            Path(f"./dev/jsons/post_prompt__{time.time_ns()}.json").write_text(json.dumps(prompt))

        body = json.dumps(apicall)
        if isinstance(body, str):
            body = body.encode()
        if post_max_size and len(body) > post_max_size:
            raise ValueError(f"Post body too large: {len(body)} bytes, post_max_size={post_max_size}")

        logger.debug("Posting prompt to url=%r, size=%d, timeout=%.1f", post_url, len(body), timeout)

        headers = { "Content-Type": "application/json" }
        resp = requests.post(post_url, data = body, headers = headers, timeout = adjusted_timeout_fn(timeout, True))
        if resp.status_code != 200:
            raise ValueError("Failed to post prompt", post_url, resp.status_code, resp.content, apicall)

        obj = resp.json()
        return obj["prompt_id"]

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

        # import logging
        # logging.getLogger("websockets").setLevel(logging.DEBUG)
        # logging.getLogger("websockets").setLevel(logging.INFO)
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

            own_prompt_id = message_obj.get("data", { }).get("prompt_id") == prompt_id
            if own_prompt_id:
                if (
                        forward_ws_messages
                        and (msg_type := message_obj.get("type")) in forwarded_types
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
                        except Exception as e:
                            logger.exception("Failed to forward remote prompt server message type=%r, node=%r, client_id=%r",
                                             msg_type, node, client_id)

                if message_obj.get("type") == "executed":
                    logger.debug("got executed message for own prompt_id=%r, size=%d", prompt_id, message_len)
                    data = message_obj["data"]
                    if data["node"] == own_id:
                        result = data["output"]
                        return result
                elif own_prompt_id and message_obj.get("type") == "execution_error":
                    raise ValueError("remote comfy prompt execution error", message_obj, base_url, prompt)
    except TimeoutError:
        # TODO: interrupt execution of that prompt in case of timeout?? always or option
        # also not sure if specific prompt interrupt is easily possible
        raise

    finally:
        if websocket is not None:
            threading.Thread(target = websocket.close, daemon = True).start()


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
        rcon, dscon = update_toggles_inplace(prompt, False, False, "input_disconnect_handler", raise_on_existing = True)
        if rcon or dscon:
            logger.info("remote_run input disconnect handler, reconnected %d, disconnected %d nodes", rcon, dscon)
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


def node_mappings(classes):
    class_map, name_map = { }, { }
    for i in classes:
        type_name = getattr(i, "TYPE_NAME", i.__name__)
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
        RemoteRunToggler,
        RemoteRunSerializerOutNode, RemoteRunDeserializerOutNode,
    )
)
