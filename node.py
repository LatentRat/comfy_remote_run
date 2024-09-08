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

import base64
import collections
import copy
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

from .node_tools import VariantSupport
from .libs.safe_load import torch_safe_load_dict
from .libs.fancy_safetensors import torch as fst_torch

import json

import logging as _logging

logger = _logging.getLogger(__name__)

jdumps = lambda data: json.dumps(data).replace("\n", "")

_NUM_OUTPUTS = 5
TOGGLE_CHOICES = ("Only Locally", "Only Remotely")


def serialization_load_input(name = "serialization"):
    return { name: (["safe_torch_pt", "unsafe_torch_pt", "fancy_safetensors"], { "default": "fancy_safetensors" }) }


def output_nodes_input(name = "output_nodes"):
    return { name: (["Keep", "Remove"], { "default": "Keep" }) }  # TODO: KeepNeeded


@VariantSupport()
class RemoteRunSerializerOutNode():
    TYPE_NAME = "RAT_RemoteRunSerializerOut"
    DISPLAY_NAME = "__Interal_Dont_Use Remote Run Serializer Output"

    CATEGORY = "Remote/Run/__Ignore__/__Internal__/"
    FUNCTION = "run"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                **serialization_load_input(),
            },
            "optional": {
                **dict((f"input_{num}", ("*", { })) for num in range(_NUM_OUTPUTS)),
            },
            "hidden":   { }
        }
        return inputs

    RETURN_TYPES = ("STRING",)

    def run(self, serialization: str, **kwargs):
        inputs = { k: v for k, v in kwargs.items() if k.startswith("input_") }

        data = serialize_obj(serialization, inputs)
        results = base64.b64encode(data).decode()
        results = (results,)

        return {
            "results": results,
            "ui":      { "results": results }
        }


@VariantSupport()
class RemoteRunDeserializerOutNode():
    TYPE_NAME = "RAT_RemoteRunDeserializerOut"
    DISPLAY_NAME = "__Interal_Dont_Use Remote Run Deserializer Output"

    CATEGORY = "Remote/Run/__Ignore__/__Internal__/"
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                **serialization_load_input(),
                "data": ("STRING", { }),
            },
        }
        return inputs

    RETURN_TYPES = tuple(["*"] * _NUM_OUTPUTS)

    def run(self, serialization: str, data: str, **kwargs):
        obj = deserialize_obj(serialization, data)
        return tuple(obj.get(i) for i in range(_NUM_OUTPUTS))


@VariantSupport()
class RemoteRunLazyToggler():
    """
    This is a Input Lazy Toggle Switch that can be used to enable/disable certain parts of a graph
    in the exact opposite way between the local and remote side using lazy evaluation.

    If the switch run_side is set to enabled it will run all inputs, otherwise set them all to off/lazy.
    When a RemoteRun runner node encounters one of these switches in a prompt it just toggles
    the run_side value so the remote side will do the exact opposite of the local one,
    without even knowing it's the remote side.
    """
    TYPE_NAME = "RAT_RemoteRunToggler"
    DISPLAY_NAME = "Remote Run Lazy Toggle"

    CATEGORY = "Remote Run"
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "enabled":              (list(TOGGLE_CHOICES), { }),
                "passthrough_when_off": ("BOOLEAN", { "default": True }),
            },
            "optional": {
                **dict((f"input_{num}", ("*", { "lazy": True })) for num in range(_NUM_OUTPUTS)),
            },
            "hidden":   {
                "dynprompt": "DYNPROMPT",
                "own_id":    "UNIQUE_ID",
            }
        }
        return inputs

    RETURN_TYPES = tuple(["*"] * _NUM_OUTPUTS)

    def check_lazy_status(self, enabled: str, dynprompt: DynamicPrompt, own_id: str, **kwargs):
        assert enabled in TOGGLE_CHOICES
        if enabled == TOGGLE_CHOICES[0]:
            # can only return names of inputs that have connections otherwise an error is thrown
            inputs = dynprompt.get_node(own_id)["inputs"]
            all_inputs = [f"input_{num}" for num in range(_NUM_OUTPUTS)]
            inputs_with_connections = [i for i in all_inputs if isinstance(inputs.get(i), list)]
            return inputs_with_connections

        return []

    def run(self, enabled: str, passthrough_when_off: bool, own_id: str, **kwargs):
        print(own_id, enabled)
        if enabled == TOGGLE_CHOICES[1] and not passthrough_when_off:
            # in case a lazy input was triggerd by something else but the value passthrough is not wanted
            return tuple(None for _ in range(_NUM_OUTPUTS))

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
        **output_nodes_input(),
    }


@VariantSupport()
class RemoteRunInputNode():
    TYPE_NAME = "RAT_RemoteRunInput"
    DISPLAY_NAME = "Remote Run Input Graph(s)"

    NUM_OUTPUTS = _NUM_OUTPUTS
    CATEGORY = "Remote Run"
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                **_shared_input_types(),
            },
            "optional": {
                **dict((f"input_{num}", ("*", { "lazy": True })) for num in range(_NUM_OUTPUTS)),
            },
            "hidden":   {
                "dynprompt": "DYNPROMPT",
                "own_id":    "UNIQUE_ID",
            }
        }
        return inputs

    RETURN_TYPES = tuple(["*"] * _NUM_OUTPUTS)

    def check_lazy_status(self, **kwargs):
        return []

    def run(self, remote_url: str, total_timeout: float, request_timeout: float, output_nodes: str, serialization: str,
            prompt = None, dynprompt: DynamicPrompt = None, own_id = None,
            **kwargs,
            ):
        start_node_ids = input_start_nodes(dynprompt, own_id)[0]
        if start_node_ids:
            # had RemoteStartNode, expand this node into a RemoteRunJson node
            return partial_json_expansion(dynprompt, own_id, remote_url, total_timeout, request_timeout, output_nodes, serialization)

        # original prompt may have been modified already, rebuild full one
        full_prompt = { i: dynprompt.get_node(i) for i in dynprompt.all_node_ids() }
        run_prompt = build_full_prompt(full_prompt, dynprompt, own_id, output_nodes)

        raw_result = remote_execute_prompt(remote_url, run_prompt, own_id, total_timeout = total_timeout, short_timeout = request_timeout)
        result_str = raw_result["results"][0]
        obj = deserialize_obj(serialization, result_str)

        result = tuple(obj.get(f"input_{num}", None) for num in range(_NUM_OUTPUTS))
        return result


@VariantSupport()
class RemoteRunJsonNode():
    TYPE_NAME = "RAT_RemoteRunJson"
    DISPLAY_NAME = "Remote Run JSON"

    NUM_OUTPUTS = _NUM_OUTPUTS
    CATEGORY = "Remote Run"
    FUNCTION = "run"

    def __init__(self):
        print("RemoteRunJsonNode.__init__ 123123123")

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                **_shared_input_types(),
                "JSON": ("STRING", { }),
            },
            "optional": {
                **dict((f"input_{num}", ("*", { })) for num in range(_NUM_OUTPUTS)),

            },
            "hidden":   {
                "prompt":    "PROMPT",
                "dynprompt": "DYNPROMPT",
                "own_id":    "UNIQUE_ID",
            }
        }
        return inputs

    RETURN_TYPES = tuple(["*"] * _NUM_OUTPUTS)

    def run(self, remote_url: str, total_timeout: float, request_timeout: float, output_nodes: str, serialization: str,
            JSON: str,
            prompt = None, dynprompt: DynamicPrompt = None, own_id = None,
            **kwargs,
            ):
        run_prompt = json.loads(JSON)
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

                    output_data_type, output_data_value = output_data
                    if output_data_type == "RunJsonInput":
                        if not output_data_value.startswith("input_") and output_data_value[6:].isdigit():
                            raise ValueError("Invalid output data", output_data, config)
                        outout_num_1start = int(output_data_value[6:])
                        data[outout_num_1start - 1] = kwargs[output_data_value]
                    elif output_data_type == "CONSTANT":
                        data[output_name] = output_data_value
                    else:
                        raise ValueError("Invalid output data", output_data, config)

                data = serialize_obj(serialization, data)
                data = base64.b64encode(data).decode()
                node["inputs"] = {
                    "serialization": serialization,
                    "data":          data,
                }

        serializers = { k: v for k, v in run_prompt.items() if v["class_type"] == RemoteRunSerializerOutNode.TYPE_NAME }
        if len(serializers) != 1:
            raise ValueError(f"expected exactly one {RemoteRunSerializerOutNode.TYPE_NAME!r} serializer node, got {len(serializers)}",
                             serializers, run_prompt)

        serialized_id = list(serializers.keys())[0]
        run_prompt = output_nodes_transform(run_prompt, output_nodes, { serialized_id })

        raw_result = remote_execute_prompt(remote_url, run_prompt, serialized_id, total_timeout = total_timeout, short_timeout = request_timeout)
        result_str = raw_result["results"][0]
        obj = deserialize_obj(serialization, result_str)

        result = tuple(obj.get(f"input_{num}", None) for num in range(_NUM_OUTPUTS))
        return result


@VariantSupport()
class RemoteRunStartNode():
    TYPE_NAME = "RAT_RemoteStart"
    DISPLAY_NAME = "Remote Run Start From Here ->"

    CATEGORY = "Remote Run"
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "optional": {
                **dict((f"input_{num}", ("*", { })) for num in range(_NUM_OUTPUTS)),
            },
        }
        return inputs

    RETURN_TYPES = tuple(["*"] * _NUM_OUTPUTS)

    def run(self, **kwargs):
        raise ValueError(f"{self.DISPLAY_NAME!r} node should always be somewhere left (up the graph) of a {RemoteRunInputNode!r}, can't run.")


def _max_id(dynprompt: DynamicPrompt):
    def toint(i):
        try:
            return int(i)
        except ValueError:
            return 0

    return max(toint(i) for i in dynprompt.all_node_ids())


def make_counter(start: int, map = None):
    def get():
        nonlocal start
        ret = start
        start += 1
        return map(ret) if map is not None else ret

    return get


def partial_json_expansion(dynprompt: DynamicPrompt, root_node_id: str,
                           remote_url: str, total_timeout: float, request_timeout: float, output_nodes: str, serialization: str):
    """
        We go from
        A -> B -> C -> RemoteStart -> D -> E -> F -> RemoteRunInput -> G -> H -> I
        to
        A -> B -> C -> RemoteRunJson(run_remotely="D -> E -> F") -> G -> H -> I

        Build an expanded graph to replace the RemoteRunInput node with a RemoteRunJson node.
        Cut out the input graph into RemoteRunInput node until a RemoteStart node is hit,
        and replace that whole chunk with a RemoteRunJson node that will run that part instead.
        The RemoteRunJson gets the same inputs as the cut-out chunks use, as it serializes them and sends them
        to the remote instance. Then gets the results back and returns them in place of the replaced RemoteRunInput node.

    """
    # TODO: output_nodes
    start_node_ids, middle_node_ids = input_start_nodes(dynprompt, root_node_id)
    assert start_node_ids

    next_node_id = make_counter(_max_id(dynprompt) + 1, str)

    # The D -> E -> F part(s), anything before the RemoteRunInput node up until to any RemoteRunStart nodes
    # (plus anything hitting the top).
    remote_prompt = { i: dynprompt.get_node(i) for i in (middle_node_ids | start_node_ids) }
    remote_prompt = copy.deepcopy(remote_prompt)

    root_node = remote_prompt.pop(root_node_id)

    # For the remote prompt: just replace each RemoteStart node with a Deserializer node that outputs
    # the serialized and sent over inputs.
    # On the local side as there can be multiple RemoteStart nodes all the inputs to the RemoteStart nodes have to also
    # be sent to the newly expanded RemoteRunJson node and kept track of which goes to which Deserializer node output.

    next_json_input_num = make_counter(1)

    local_expanded_inputs = { }
    for start_id in start_node_ids:
        start_node = remote_prompt[start_id]
        start_inputs = start_node["inputs"]

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
        remote_prompt[start_id] = {
            "class_type":          RemoteRunDeserializerOutNode.TYPE_NAME,
            "deserializer_config": {
                "outputs": des,
            },
        }

    root_inputs = { k: v for k, v in root_node["inputs"].items() if isinstance(v, list) }
    remote_prompt[root_node_id] = {
        "class_type": RemoteRunSerializerOutNode.TYPE_NAME,
        "inputs":     {
            "serialization": serialization,
            **root_inputs,
        },
    }

    json_node_id_str = next_node_id()
    new_graph = {
        json_node_id_str: {
            "class_type": RemoteRunJsonNode.TYPE_NAME,
            "inputs":     {
                "JSON": json.dumps(remote_prompt),
                **dict(
                    remote_url = remote_url,
                    total_timeout = total_timeout,
                    request_timeout = request_timeout,
                    output_nodes = output_nodes,
                    serialization = serialization,
                ),
                **local_expanded_inputs,
            },
        }
    }
    toggle_switches_inplace(new_graph)
    new_graph = output_nodes_transform(new_graph, output_nodes, { root_node_id })
    # print("new_graph", json.dumps(new_graph, indent = 4))

    replaced_output_ids = tuple([json_node_id_str, i] for i in range(_NUM_OUTPUTS))
    return {
        "result": replaced_output_ids,
        "expand": new_graph,
    }


def input_start_nodes(dynprompt: DynamicPrompt, start_id: str):
    start_nodes = set()
    middle_node_ids = set()

    def _nodes(node_id: str):
        if node_id in middle_node_ids:
            return
        middle_node_ids.add(node_id)

        node = dynprompt.get_node(node_id)
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


def deserialize_obj(serialization: str, result_str: str):
    data = base64.b64decode(result_str)
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


def build_full_prompt(prompt: dict, dynprompt: DynamicPrompt, start_id: str, output_nodes: str) -> dict:
    prompt = copy.deepcopy(prompt)
    own_node = dynprompt.get_node(start_id)

    # simple = True
    simple = False
    if simple:
        # Simple method: Copy full current prompt and just delete all outputs from it,
        # then replace own node with an input serializing & returning node.
        # Only the input nodes to this should be run in theory unless any extension/node does anything weird.

        run_prompt = copy.deepcopy(prompt)
        output_ids = prompt_outputs(run_prompt)
        run_prompt = { k: v for k, v in run_prompt.items() if k not in output_ids }
    else:
        # Go though input node graphs and make new prompt only from nodes that lead to the inputs of this node.
        input_nodes = set()

        def _nodes(node_id: str, seen):
            if node_id in seen:
                return
            seen.add(node_id)

            node = dynprompt.get_node(node_id)
            inputs = node["inputs"]
            for inp in inputs.values():
                if isinstance(inp, list):
                    input_nodes.add(inp[0])
                    _nodes(inp[0], seen)

        _nodes(start_id, set())
        run_prompt = { i: dynprompt.get_node(i) for i in input_nodes }
        run_prompt = copy.deepcopy(run_prompt)

    inputs = copy.deepcopy(own_node["inputs"])
    inputs = { k: v for k, v in inputs.items() if k.startswith("input_") or k == "serialization" }

    toggle_switches_inplace(run_prompt)

    # TODO: option to only keep outputs needed for this node's inputs
    run_prompt = output_nodes_transform(run_prompt, output_nodes)

    # replace own node with input serializer output node
    run_prompt[start_id] = {
        "inputs":     inputs,
        "class_type": RemoteRunSerializerOutNode.TYPE_NAME,
    }

    return run_prompt


def toggle_switches_inplace(prompt: dict):
    for node_id, node in prompt.items():
        if node["class_type"] == RemoteRunLazyToggler.TYPE_NAME:
            assert node["inputs"]["enabled"] in TOGGLE_CHOICES
            node["inputs"]["enabled"] = TOGGLE_CHOICES[1] if node["inputs"]["enabled"] == TOGGLE_CHOICES[0] else TOGGLE_CHOICES[0]


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


def remote_execute_prompt(base_url: str, prompt: dict, own_id: str, total_timeout: float, short_timeout: float, total_tries = 3):
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
        )
        resp = requests.post(post_url, json = apicall, timeout = adjusted_timeout_fn(timeout, True))
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
                max_size = 64 * 1024 ** 2,
            )

        websocket = connect()
        prompt_id = post(short_timeout)

        while True:
            message = websocket.recv(timeout = adjusted_timeout_fn(total_timeout, True))
            message = json.loads(message)

            own_prompt_id = message.get("data", { }).get("prompt_id") == prompt_id
            if own_prompt_id and message.get("type") == "executed":
                data = message["data"]
                if data["node"] == own_id:
                    result = data["output"]
                    return result
            elif own_prompt_id and message.get("type") == "execution_error":
                raise ValueError("remote comfy prompt execution error", message, base_url, prompt)

    except TimeoutError:
        # TODO: interrupt execution of that prompt in case of timeout?? always or option
        # also not sure if specific prompt interrupt is easily possible
        raise

    finally:
        if websocket is not None:
            threading.Thread(target = websocket.close, daemon = True).start()


def add_server_prompt_handler():
    logger.info("Adding promptserver remote run nonce no duplicate prompt handler")
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

    if True:
        # for hotreload
        to_del = [i for i in server.PromptServer.instance.on_prompt_handlers if getattr(i, "rr_nonce_nodup_clearable", False)]
        for i in to_del:
            print("clearing old remote run nonce prompt handler", i)
            server.PromptServer.instance.on_prompt_handlers.remove(i)
        prompt_handler.rr_nonce_nodup_clearable = True

    server.PromptServer.instance.add_on_prompt_handler(prompt_handler)


if os.environ.get("SKIP_REMOTE_RUN_NONCE_HANDLER") == "1":
    logger.info("SKIP_REMOTE_RUN_NONCE_HANDLER=1, skipping adding remote run no duplicate nonce handler!")
else:
    logger.info("adding remote run no duplicate nonce handler for!")
    add_server_prompt_handler()


def node_mappings(classes):
    class_map, name_map = { }, { }
    for i in classes:
        type_name = getattr(i, "TYPE_NAME", i.__name__)
        class_map[type_name] = i
        name_map[type_name] = getattr(i, "DISPLAY_NAME", type_name)

    return class_map, name_map


NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = node_mappings(
    (
        RemoteRunStartNode, RemoteRunInputNode, RemoteRunLazyToggler, RemoteRunJsonNode,
        RemoteRunSerializerOutNode, RemoteRunDeserializerOutNode,
    )
)
