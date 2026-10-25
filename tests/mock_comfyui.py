#!/usr/bin/env python3
"""Headless smoke tests for the ComfyUI Newelle extension.

Runs the extension's handler against a stub ComfyUI HTTP server without
Newelle or GTK installed: the Newelle modules are replaced with minimal
fakes that mimic the real settings API, exactly the way Newelle's
ExtensionLoader imports the file (as part of the ``newelle`` package).

Usage:
    python tests/mock_comfyui.py
"""

import copy
import importlib.util
import io
import json
import os
import socket
import sys
import tempfile
import threading
import time
import types
import unittest
from enum import Enum
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTENSION_PATH = os.path.join(REPO, "comfyui.py")

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

# Stored settings shared by most tests: a cached model list so the handler
# constructor does not start a background fetch.
CACHED_MODELS = json.dumps(
    [["modelA.safetensors", "modelA.safetensors"]]
)


# ----------------------------------------------------------------------
# Fake Newelle modules (installed before the extension is imported)
# ----------------------------------------------------------------------


class ErrorSeverity(Enum):
    NONE = 0
    WARNING = 1
    ERROR = 2


class _FakeExtraSettings:
    """Same dict shapes as Newelle's real ExtraSettings factories."""

    @staticmethod
    def _setting(key, title, description, default, **kwargs):
        row = {
            "key": key,
            "title": title,
            "description": description,
            "default": default,
            "update_settings": kwargs.get("update_settings", False),
        }
        for name in ("website", "folder", "refresh", "refresh_icon"):
            if name in kwargs:
                row[name] = kwargs[name]
        return row

    @classmethod
    def EntrySetting(cls, key, title, description, default, **kwargs):
        row = cls._setting(key, title, description, default, **kwargs)
        row["type"] = "entry"
        row["password"] = kwargs.get("password", False)
        return row

    @classmethod
    def MultilineEntrySetting(cls, key, title, description, default, **kwargs):
        row = cls._setting(key, title, description, default, **kwargs)
        row["type"] = "multilineentry"
        return row

    @classmethod
    def ToggleSetting(cls, key, title, description, default, **kwargs):
        row = cls._setting(key, title, description, default, **kwargs)
        row["type"] = "toggle"
        return row

    @classmethod
    def NestedSetting(cls, key, title, description, extra_settings, **kwargs):
        row = cls._setting(key, title, description, None, **kwargs)
        row["type"] = "nested"
        row["extra_settings"] = extra_settings
        return row

    @classmethod
    def ButtonSetting(cls, key, title, description, callback, label=None, **kwargs):
        row = cls._setting(key, title, description, None, **kwargs)
        row["type"] = "button"
        row["callback"] = callback
        if label is not None:
            row["label"] = label
        return row

    @classmethod
    def ComboSetting(cls, key, title, description, values, default, **kwargs):
        row = cls._setting(key, title, description, default, **kwargs)
        row["type"] = "combo"
        row["values"] = values
        return row

    @classmethod
    def SpinSetting(
        cls, key, title, description, default, minimum, maximum, **kwargs
    ):
        row = cls._setting(key, title, description, default, **kwargs)
        row["type"] = "spin"
        row["min"] = minimum
        row["max"] = maximum
        return row


def _HandlerDescription(key, title, description, handler_class, website=None,
                        primary=True, secondary=True, wakeword=False):
    return {
        "key": key,
        "title": title,
        "description": description,
        "class": handler_class,
        "primary": primary,
        "secondary": secondary,
        "wakeword": wakeword,
        "website": website,
    }


class _FakeHandlerBase:
    """Minimal stand-in for Newelle's Handler + ImageGeneratorHandler."""

    key = ""
    schema_key = "image-generator-settings"

    def __init__(self, settings, path):
        self.settings = settings
        self.path = path
        self.pip_path = path
        self._values = dict(settings) if isinstance(settings, dict) else {}
        self._thrown = []
        self.on_extra_settings_update = None
        self.cache_dir = os.path.join(path, "generated_images")
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_setting(self, key, search_default=True, return_value=None):
        if key in self._values:
            return self._values[key]
        if search_default:
            return self.get_default_setting(key)
        return return_value

    def set_setting(self, key, value):
        self._values[key] = value

    def get_default_setting(self, key):
        for setting in self.get_extra_settings():
            if setting.get("type") == "nested":
                for child in setting.get("extra_settings", []):
                    if child["key"] == key:
                        return child.get("default")
            if setting["key"] == key:
                return setting.get("default")
        return None

    def settings_update(self):
        pass

    def throw(self, message, severity=ErrorSeverity.WARNING):
        self._thrown.append((message, severity))

    def cache_path_for(self, msg_uuid):
        return os.path.join(self.cache_dir, f"{msg_uuid}.png")

    def get_tools(self):
        return [_FakeTool("generate_image", self._generate_image_tool)]

    def generate_and_display(self, prompt, widget, msg_uuid,
                             on_done_callback=None):
        """Stand-in mirroring Newelle 1.5's worker flow.

        The real base class swallows generation errors (it only prints
        them), has no on_error_callback parameter, and invokes
        on_done_callback on success and failure alike. The worker thread
        is omitted here to keep the tests deterministic.
        """
        output_path = self.cache_path_for(msg_uuid)
        try:
            image_source = self.generate_image(prompt, msg_uuid, output_file=output_path)
            if not image_source:
                print("Image generation failed: no image returned")
        except Exception as error:  # noqa: BLE001 - the base swallows errors too
            print(f"Image generation failed: {error}")
        if on_done_callback:
            on_done_callback()


class _FakeNewelleExtension:
    id = "demoextension"
    name = "Demo Extension"

    def __init__(self, pip_path, extension_path, settings):
        self.pip_path = pip_path
        self.path = pip_path
        self.extension_path = extension_path
        self.settings = settings

    def get_image_generator_handlers(self):
        return []


class _FakeTool:
    """Minimal stand-in for Newelle's Tool."""

    def __init__(self, name, func, restore_func=None, default_on=False):
        self.name = name
        self.func = func
        self.restore_func = restore_func
        self.default_on = default_on


class _FakeToolResult:
    """Semaphore-faithful copy of Newelle's ToolResult."""

    def __init__(self, output=None, widget=None, requires_interaction=False,
                 interaction_options=None, display_text=None, context_messages=None,
                 stop_generation=False, cancel_callback=None):
        import threading

        self.output = output
        self.widget = widget
        self.display_text = display_text
        self.context_messages = list(context_messages or [])
        self.stop_generation = bool(stop_generation)
        self.cancel_callback = cancel_callback
        self.is_cancelled = False
        self.requires_interaction = requires_interaction
        self.interaction_options = interaction_options or []
        self.output_semaphore = threading.Semaphore()
        self.output_semaphore.acquire()

    def get_output(self):
        self.output_semaphore.acquire()
        self.output_semaphore.release()
        return self.output

    def set_output(self, output):
        self.output = output
        try:
            self.output_semaphore.release()
        except ValueError:
            # Semaphore already released
            pass

    def set_widget(self, widget):
        self.widget = widget

    def get_context_messages(self):
        return list(self.context_messages)

    def cancel(self):
        self.is_cancelled = True
        self.set_output(None)


class _FakeImageGeneratorWidget:
    def __init__(self, width=400, height=400):
        self.width = width
        self.height = height
        self.prompt = None

    def set_prompt(self, prompt):
        self.prompt = prompt


def load_extension_module():
    """Import comfyui.py as ``newelle.comfyui`` with the fakes installed."""
    if "newelle.comfyui" in sys.modules:
        return sys.modules["newelle.comfyui"]

    newelle = types.ModuleType("newelle")
    newelle.__path__ = []
    handlers = types.ModuleType("newelle.handlers")
    handlers.ErrorSeverity = ErrorSeverity
    handlers.ExtraSettings = _FakeExtraSettings
    handlers.HandlerDescription = _HandlerDescription
    handlers.ImageGeneratorHandler = _FakeHandlerBase

    tools_mod = types.ModuleType("newelle.tools")
    tools_mod.Tool = _FakeTool
    tools_mod.ToolResult = _FakeToolResult

    ui_mod = types.ModuleType("newelle.ui")
    widgets_mod = types.ModuleType("newelle.ui.widgets")
    image_gen_mod = types.ModuleType("newelle.ui.widgets.image_generator")
    image_gen_mod.ImageGeneratorWidget = _FakeImageGeneratorWidget

    extensions = types.ModuleType("newelle.extensions")
    extensions.NewelleExtension = _FakeNewelleExtension

    sys.modules["newelle"] = newelle
    sys.modules["newelle.handlers"] = handlers
    sys.modules["newelle.tools"] = tools_mod
    sys.modules["newelle.ui"] = ui_mod
    sys.modules["newelle.ui.widgets"] = widgets_mod
    sys.modules["newelle.ui.widgets.image_generator"] = image_gen_mod
    sys.modules["newelle.extensions"] = extensions
    newelle.handlers = handlers
    newelle.tools = tools_mod
    newelle.ui = ui_mod
    newelle.extensions = extensions

    if "gi.repository" not in sys.modules:
        gi = types.ModuleType("gi")
        gi_repository = types.ModuleType("gi.repository")

        class _GLib:
            @staticmethod
            def idle_add(function, *args):
                function(*args)
                return 0

        gi_repository.GLib = _GLib
        gi.repository = gi_repository
        sys.modules["gi"] = gi
        sys.modules["gi.repository"] = gi_repository

    spec = importlib.util.spec_from_file_location("newelle.comfyui", EXTENSION_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["newelle.comfyui"] = module
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------------
# Stub ComfyUI server
# ----------------------------------------------------------------------


def make_png():
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (200, 30, 30)).save(buffer, "PNG")
    return buffer.getvalue()


class MockComfyUI:
    """Minimal ComfyUI HTTP API: /object_info, /prompt, /history, /view."""

    def __init__(self, delay=0.0, fail=False, lose=False):
        self.delay = delay
        self.fail = fail
        self.lose = lose  # simulate a server restart: prompt vanishes
        self.png = make_png()
        self.prompts = []  # list of (prompt_id, workflow, client_id)
        self.views = []  # list of parsed query dicts
        self.lock = threading.Lock()

        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _json(self, data, code=200):
                body = json.dumps(data).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                parsed = urlparse(self.path)
                if parsed.path == "/object_info":
                    self._json(
                        {
                            "CheckpointLoaderSimple": {
                                "input": {
                                    "required": {
                                        "ckpt_name": [
                                            ["modelA.safetensors", "modelB.safetensors"],
                                            {},
                                        ]
                                    }
                                }
                            }
                        }
                    )
                elif parsed.path.startswith("/history/"):
                    prompt_id = parsed.path.rsplit("/", 1)[-1]
                    self._json(outer.get_history(prompt_id) or {})
                elif parsed.path == "/queue":
                    self._json(outer.get_queue_state())
                elif parsed.path == "/view":
                    outer.views.append(parse_qs(parsed.query))
                    body = outer.png
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self._json({"error": "unknown endpoint"}, 404)

            def do_POST(self):
                if urlparse(self.path).path == "/prompt":
                    length = int(self.headers.get("Content-Length", 0))
                    body = json.loads(self.rfile.read(length))
                    prompt_id = outer.submit(
                        body.get("prompt"), body.get("client_id")
                    )
                    self._json({"prompt_id": prompt_id})
                else:
                    self._json({"error": "unknown endpoint"}, 404)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def submit(self, workflow, client_id):
        prompt_id = f"prompt-{len(self.prompts) + 1}"
        with self.lock:
            self.prompts.append(
                {
                    "id": prompt_id,
                    "workflow": workflow,
                    "client_id": client_id,
                    "ready_at": time.time() + self.delay,
                }
            )
        return prompt_id

    def get_history(self, prompt_id):
        if self.lose:
            return None
        with self.lock:
            for prompt in self.prompts:
                if prompt["id"] == prompt_id:
                    if time.time() < prompt["ready_at"]:
                        return {}
                    if self.fail:
                        return {
                            prompt_id: {
                                "status": {
                                    "status_str": "error",
                                    "messages": [
                                        [
                                            "execution_error",
                                            {
                                                "exception_type": "OSError",
                                                "exception_message": "boom",
                                                "node_id": "3",
                                                "node_type": "KSampler",
                                            },
                                        ]
                                    ],
                                },
                                "outputs": {},
                            }
                        }
                    return {
                        prompt_id: {
                            "status": {"status_str": "success", "messages": []},
                            "outputs": {
                                "9": {
                                    "images": [
                                        {
                                            "filename": "ComfyUI_00001_.png",
                                            "subfolder": "",
                                            "type": "output",
                                        }
                                    ]
                                }
                            },
                        }
                    }
        return None

    def get_queue_state(self):
        if self.lose:
            return {"queue_running": [], "queue_pending": []}
        now = time.time()
        pending = []
        with self.lock:
            for index, prompt in enumerate(self.prompts):
                if now < prompt["ready_at"]:
                    pending.append([index, prompt["id"], {}, {}, []])
        return {"queue_running": [], "queue_pending": pending}

    def last_workflow(self):
        with self.lock:
            return self.prompts[-1]["workflow"]

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


def free_port():
    """Grab a port that is (almost certainly) closed."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

# A custom workflow exported in API format, with KSamplerAdvanced
# (noise_seed), non-sequential node IDs and a different checkpoint node.
CUSTOM_WORKFLOW = {
    "10": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "", "clip": ["11", 1]},
    },
    "11": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "placeholder.safetensors"},
    },
    "12": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "", "clip": ["11", 1]},
    },
    "13": {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": 640, "height": 960, "batch_size": 1},
    },
    "14": {
        "class_type": "KSamplerAdvanced",
        "inputs": {
            "add_noise": "enable",
            "noise_seed": 7,
            "steps": 25,
            "cfg": 3.5,
            "sampler_name": "euler",
            "scheduler": "normal",
            "start_at_step": 0,
            "end_at_step": 10000,
            "return_with_leftover_noise": "disable",
            "model": ["11", 0],
            "positive": ["10", 0],
            "negative": ["12", 0],
            "latent_image": ["13", 0],
        },
    },
    "15": {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["14", 0], "vae": ["11", 2]},
    },
    "16": {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": "newelle", "images": ["15", 0]},
    },
}


# A Qwen-Image / Wan style workflow: one combined text encoder holding
# both "prompt" and "negative_prompt", wired to both sampler slots.
QWEN_COMBINED_WORKFLOW = {
    "10": {
        "class_type": "TextEncodeQwenImage21",
        "inputs": {
            "prompt": "",
            "negative_prompt": "",
            "clip": ["11", 0],
        },
    },
    "11": {
        "class_type": "CLIPLoader",
        "inputs": {
            "clip_name": "qwen3vl_4b_fp8_scaled.safetensors",
            "type": "qwen_image",
        },
    },
    "12": {
        "class_type": "UNETLoader",
        "inputs": {"unet_name": "placeholder.safetensors", "weight_dtype": "default"},
    },
    "13": {
        "class_type": "EmptySD3LatentImage",
        "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
    },
    "14": {
        "class_type": "KSampler",
        "inputs": {
            "seed": 1,
            "steps": 20,
            "cfg": 4.0,
            "sampler_name": "res",
            "scheduler": "normal",
            "denoise": 1.0,
            "model": ["12", 0],
            "positive": ["10", 0],
            "negative": ["10", 1],
            "latent_image": ["13", 0],
        },
    },
    "15": {
        "class_type": "VAELoader",
        "inputs": {"vae_name": "ae.safetensors"},
    },
    "16": {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["14", 0], "vae": ["15", 0]},
    },
    "17": {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": "newelle", "images": ["16", 0]},
    },
}


# An SD3-style workflow with multi-CLIP encoders exposing one text input
# per text model.
MULTI_CLIP_WORKFLOW = {
    "20": {
        "class_type": "CLIPTextEncodeSD3",
        "inputs": {"clip_l": "", "clip_g": "", "t5xxl": "", "clip": ["22", 0]},
    },
    "21": {
        "class_type": "CLIPTextEncodeSD3",
        "inputs": {"clip_l": "", "clip_g": "", "t5xxl": "", "clip": ["22", 0]},
    },
    "22": {
        "class_type": "TripleCLIPLoader",
        "inputs": {
            "clip_name1": "a.safetensors",
            "clip_name2": "b.safetensors",
            "clip_name3": "c.safetensors",
        },
    },
    "23": {
        "class_type": "EmptySD3LatentImage",
        "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
    },
    "24": {
        "class_type": "KSampler",
        "inputs": {
            "seed": 1,
            "steps": 28,
            "cfg": 4.5,
            "sampler_name": "dpmpp_2m",
            "scheduler": "sgm_uniform",
            "denoise": 1.0,
            "model": ["25", 0],
            "positive": ["20", 0],
            "negative": ["21", 0],
            "latent_image": ["23", 0],
        },
    },
    "25": {
        "class_type": "UNETLoader",
        "inputs": {"unet_name": "placeholder.safetensors", "weight_dtype": "default"},
    },
    "26": {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["24", 0], "vae": ["27", 0]},
    },
    "27": {
        "class_type": "VAELoader",
        "inputs": {"vae_name": "ae.safetensors"},
    },
    "28": {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": "newelle", "images": ["26", 0]},
    },
}


class ComfyUIExtensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_extension_module()
        cls.module.POLL_INTERVAL = 0.05
        cls.server = MockComfyUI(delay=0.0)

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()

    def make_handler(self, overrides=None):
        overrides = dict(overrides or {})
        overrides.setdefault("url", self.server.url)
        overrides.setdefault("models", CACHED_MODELS)
        directory = tempfile.mkdtemp(prefix="newelle-comfyui-test-")
        return self.module.ComfyUIHandler(overrides, directory)

    def assert_png(self, path):
        self.assertTrue(os.path.isfile(path), f"missing file: {path}")
        with open(path, "rb") as file:
            self.assertEqual(file.read(len(PNG_MAGIC)), PNG_MAGIC)

    # -- registration ---------------------------------------------------

    def test_extension_registers_handler(self):
        extension = self.module.ComfyUIExtension("/tmp/pip", "/tmp/cache", {})
        handlers = extension.get_image_generator_handlers()
        self.assertEqual(len(handlers), 1)
        self.assertEqual(handlers[0]["key"], "comfyui")
        self.assertEqual(handlers[0]["title"], "ComfyUI")
        self.assertIs(handlers[0]["class"], self.module.ComfyUIHandler)
        self.assertEqual(self.module.ComfyUIHandler.key, "comfyui")

    # -- generation with the default workflow ----------------------------

    def test_default_workflow_generation(self):
        handler = self.make_handler({"model": "modelB.safetensors"})
        path = handler.generate_image("a cat", "gen-1")
        self.assert_png(path)
        self.assertEqual(path, handler.cache_path_for("gen-1"))

        workflow = self.server.last_workflow()
        self.assertEqual(workflow["6"]["inputs"]["text"], "a cat")
        self.assertEqual(workflow["4"]["inputs"]["ckpt_name"], "modelB.safetensors")
        self.assertEqual(workflow["5"]["inputs"]["width"], 512)
        self.assertEqual(workflow["5"]["inputs"]["height"], 512)
        self.assertEqual(workflow["5"]["inputs"]["batch_size"], 1)
        self.assertEqual(workflow["3"]["inputs"]["steps"], 20)
        self.assertIsInstance(workflow["3"]["inputs"]["seed"], int)
        # Empty negative prompt is not injected: keeps the exported value.
        self.assertEqual(workflow["7"]["inputs"]["text"], "")

        self.assertTrue(self.server.views)
        view = self.server.views[-1]
        self.assertEqual(view["filename"], ["ComfyUI_00001_.png"])
        self.assertEqual(view["type"], ["output"])

    def test_negative_prompt_and_template(self):
        handler = self.make_handler(
            {
                "negative-prompt": "blurry, lowres",
                "prompt-template": "masterpiece, [input]",
                "model": "modelA.safetensors",
            }
        )
        path = handler.generate_image("a cat", "gen-2")
        self.assert_png(path)
        workflow = self.server.last_workflow()
        self.assertEqual(workflow["6"]["inputs"]["text"], "masterpiece, a cat")
        self.assertEqual(workflow["7"]["inputs"]["text"], "blurry, lowres")

    def test_fixed_seed(self):
        handler = self.make_handler({"random-seed": False, "seed": 123456})
        handler.generate_image("a cat", "gen-3")
        workflow = self.server.last_workflow()
        self.assertEqual(workflow["3"]["inputs"]["seed"], 123456)

    # -- custom workflow, auto-detection and mapping ---------------------

    def test_custom_workflow_autodetect(self):
        handler = self.make_handler(
            {
                "workflow": json.dumps(CUSTOM_WORKFLOW),
                "model": "modelA.safetensors",
                "negative-prompt": "watermark",
                "width": 768,
                "height": 768,
            }
        )
        handler.generate_image("a dog", "gen-4")
        workflow = self.server.last_workflow()
        # Prompt/negative resolved through the sampler's positive/negative.
        self.assertEqual(workflow["10"]["inputs"]["text"], "a dog")
        self.assertEqual(workflow["12"]["inputs"]["text"], "watermark")
        # Model, size and steps auto-detected.
        self.assertEqual(workflow["11"]["inputs"]["ckpt_name"], "modelA.safetensors")
        self.assertEqual(workflow["13"]["inputs"]["width"], 768)
        self.assertEqual(workflow["13"]["inputs"]["height"], 768)
        self.assertEqual(workflow["14"]["inputs"]["steps"], 20)
        # KSamplerAdvanced uses noise_seed, detected automatically.
        self.assertIsInstance(workflow["14"]["inputs"]["noise_seed"], int)

    def test_manual_node_mapping_and_custom_values(self):
        handler = self.make_handler(
            {
                "workflow": json.dumps(CUSTOM_WORKFLOW),
                "prompt-node": "10",
                "seed-node": "14",
                "custom-values": '{"14": {"cfg": 2.5}}',
                "model": "modelA.safetensors",
            }
        )
        handler.generate_image("a dog", "gen-5")
        workflow = self.server.last_workflow()
        self.assertEqual(workflow["10"]["inputs"]["text"], "a dog")
        self.assertEqual(workflow["14"]["inputs"]["cfg"], 2.5)

    def test_combined_text_encoder(self):
        """Qwen/Wan encoders keep prompt and negative_prompt on one node."""
        handler = self.make_handler(
            {
                "workflow": json.dumps(QWEN_COMBINED_WORKFLOW),
                "model": "Qwen-Image-2.1-Turbo-INT8-ConvRot.safetensors",
                "negative-prompt": "watermark",
                "width": 832,
                "height": 832,
            }
        )
        handler.generate_image("a dog", "gen-qw")
        workflow = self.server.last_workflow()
        self.assertEqual(workflow["10"]["inputs"]["prompt"], "a dog")
        self.assertEqual(workflow["10"]["inputs"]["negative_prompt"], "watermark")
        # The model landed in the UNETLoader's *_name input, and size in
        # the SD3 latent node.
        self.assertEqual(
            workflow["12"]["inputs"]["unet_name"],
            "Qwen-Image-2.1-Turbo-INT8-ConvRot.safetensors",
        )
        self.assertEqual(workflow["13"]["inputs"]["width"], 832)
        self.assertEqual(workflow["13"]["inputs"]["height"], 832)

    def test_multiclip_encoder(self):
        """SD3/Flux-style encoders receive the prompt on every text input."""
        handler = self.make_handler(
            {
                "workflow": json.dumps(MULTI_CLIP_WORKFLOW),
                "model": "flux-dev.safetensors",
                "negative-prompt": "blurry",
            }
        )
        handler.generate_image("a dog", "gen-sd3")
        workflow = self.server.last_workflow()
        for key in ("clip_l", "clip_g", "t5xxl"):
            self.assertEqual(workflow["20"]["inputs"][key], "a dog")
            self.assertEqual(workflow["21"]["inputs"][key], "blurry")
        # The loader triple is not touched: its keys are not text inputs.
        self.assertEqual(
            workflow["22"]["inputs"]["clip_name1"], "a.safetensors"
        )

    def test_negative_falls_back_to_prompt_key(self):
        """Encoders with only a 'prompt' input work as negative node."""
        workflow = copy.deepcopy(QWEN_COMBINED_WORKFLOW)
        # Rewire the negative slot to a separate single-prompt encoder.
        workflow["18"] = {
            "class_type": "TextEncodeQwenImageEdit",
            "inputs": {"prompt": "", "clip": ["11", 0]},
        }
        workflow["14"]["inputs"]["negative"] = ["18", 0]
        handler = self.make_handler(
            {
                "workflow": json.dumps(workflow),
                "model": "placeholder.safetensors",
                "negative-prompt": "lowres",
            }
        )
        handler.generate_image("a dog", "gen-neg")
        result = self.server.last_workflow()
        self.assertEqual(result["10"]["inputs"]["prompt"], "a dog")
        self.assertEqual(result["18"]["inputs"]["prompt"], "lowres")

    def test_model_is_optional(self):
        """Without a model selection, the workflow's model is kept."""
        handler = self.make_handler()
        path = handler.generate_image("a cat", "gen-opt")
        self.assert_png(path)
        workflow = self.server.last_workflow()
        self.assertEqual(
            workflow["4"]["inputs"]["ckpt_name"],
            "v1-5-pruned-emaonly-fp16.safetensors",
        )

    def test_input_key_overrides(self):
        """Per-role input keys beat automatic key detection."""
        handler = self.make_handler(
            {
                "workflow": json.dumps(QWEN_COMBINED_WORKFLOW),
                "model": "modelA.safetensors",
                "model-key": "unet_name",
                "prompt-node": "10",
                "prompt-key": "text",
                "negative-node": "10",
                "negative-key": "negative_prompt",
                "negative-prompt": "watermark",
                "seed-key": "noise_seed",
            }
        )
        handler.generate_image("a dog", "gen-keys")
        workflow = self.server.last_workflow()
        # "text" does not exist on TextEncodeQwenImage21: an explicit key
        # is applied anyway, and the automatic "prompt" key is not used.
        self.assertEqual(workflow["10"]["inputs"]["text"], "a dog")
        self.assertEqual(workflow["10"]["inputs"]["prompt"], "")
        self.assertEqual(workflow["10"]["inputs"]["negative_prompt"], "watermark")
        self.assertEqual(workflow["12"]["inputs"]["unet_name"], "modelA.safetensors")
        self.assertIsInstance(workflow["14"]["inputs"]["noise_seed"], int)

    def test_import_workflow_validation(self):
        handler = self.make_handler()
        # API-format workflows pass validation.
        handler._validate_workflow_text(json.dumps(CUSTOM_WORKFLOW))
        handler._validate_workflow_text(json.dumps(QWEN_COMBINED_WORKFLOW))
        # Not JSON at all.
        with self.assertRaises(RuntimeError) as context:
            handler._validate_workflow_text("<html>not json</html>")
        self.assertIn("not valid JSON", str(context.exception))
        # UI-format export is recognized and rejected with a hint.
        with self.assertRaises(RuntimeError) as context:
            handler._validate_workflow_text(
                json.dumps({"nodes": [], "links": [], "last_node_id": 0})
            )
        self.assertIn("(UI) format", str(context.exception))
        # Wrong node shape.
        with self.assertRaises(RuntimeError) as context:
            handler._validate_workflow_text(
                json.dumps({"1": {"class_type": "KSampler"}})
            )
        self.assertIn("not in ComfyUI API format", str(context.exception))
        # Not an object.
        with self.assertRaises(RuntimeError):
            handler._validate_workflow_text(json.dumps([1, 2, 3]))

    def test_ui_format_paste_rejected(self):
        handler = self.make_handler(
            {"workflow": json.dumps({"nodes": [], "links": []})}
        )
        with self.assertRaises(RuntimeError) as context:
            handler.generate_image("a dog", "gen-ui")
        self.assertIn("(UI) format", str(context.exception))

    def test_prompt_lost_detected(self):
        """A prompt vanishing from queue and history fails fast."""
        server = MockComfyUI(lose=True)
        try:
            handler = self.make_handler({"url": server.url})
            with self.assertRaises(RuntimeError) as context:
                handler.generate_image("a cat", "gen-lost")
            self.assertIn("disappeared", str(context.exception))
        finally:
            server.stop()

    def test_tool_reports_text_output(self):
        """The image tool releases its ToolResult with a textual output.

        Newelle 1.5's default image tool never releases its result, which
        blocks the generation chain and leaves the chat's stop button
        visible after the image is done. A textual output closes the
        turn like every other tool, and the widget is still shown.
        """
        handler = self.make_handler()
        tools = handler.get_tools()
        tool = next(item for item in tools if item.name == "generate_image")
        self.assertIs(tool.func.__func__, handler._generate_image_tool.__func__)

        result = tool.func("a cat", "msg-1")
        output = result.get_output()  # must not block
        self.assertIn("generated successfully", str(output))
        self.assertIsNotNone(result.widget)
        self.assertEqual(handler._generation_errors, {})

    def test_tool_reports_error_output(self):
        """Generation failures release the result with an error output.

        Newelle 1.5's worker only ever calls on_done_callback (errors are
        printed, not passed on), so the outcome recorded by
        generate_image decides what the tool reports.
        """
        handler = self.make_handler({"url": f"http://127.0.0.1:{free_port()}"})
        tools = handler.get_tools()
        tool = next(item for item in tools if item.name == "generate_image")
        result = tool.func("a cat", "msg-2")
        output = result.get_output()  # must not block
        self.assertIn("Image generation failed", str(output))
        self.assertIsNotNone(result.widget)
        self.assertEqual(handler._generation_errors, {})

    def test_generate_image_tracks_outcome_per_message(self):
        """generate_image records the failure and clears it on success."""
        handler = self.make_handler({"url": f"http://127.0.0.1:{free_port()}"})
        with self.assertRaises(RuntimeError):
            handler.generate_image("a cat", "msg-3")
        with handler._generation_error_lock:
            self.assertIn("msg-3", handler._generation_errors)

        working = self.make_handler()
        path = working.generate_image("a cat", "msg-4")
        self.assertTrue(os.path.exists(path))
        with working._generation_error_lock:
            self.assertEqual(working._generation_errors, {})

    def test_unknown_node_raises(self):
        handler = self.make_handler(
            {
                "workflow": json.dumps(CUSTOM_WORKFLOW),
                "prompt-node": "99",
            }
        )
        with self.assertRaises(RuntimeError) as context:
            handler.generate_image("a dog", "gen-6")
        self.assertIn("was not found in the workflow", str(context.exception))

    def test_no_prompt_node_raises(self):
        workflow = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "x.safetensors"},
            }
        }
        handler = self.make_handler({"workflow": json.dumps(workflow)})
        with self.assertRaises(RuntimeError) as context:
            handler.generate_image("a dog", "gen-7")
        self.assertIn("No text prompt node", str(context.exception))

    def test_model_without_loader_raises(self):
        workflow = {
            "10": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "", "clip": ["11", 1]},
            },
            "11": {
                "class_type": "CLIPLoader",
                "inputs": {"clip_name": "x.safetensors"},
            },
        }
        handler = self.make_handler(
            {"workflow": json.dumps(workflow), "model": "modelA.safetensors"}
        )
        with self.assertRaises(RuntimeError) as context:
            handler.generate_image("a dog", "gen-8")
        self.assertIn("no checkpoint or UNET loader", str(context.exception))

    def test_invalid_workflow_json(self):
        handler = self.make_handler({"workflow": "{not json"})
        with self.assertRaises(RuntimeError) as context:
            handler.generate_image("a dog", "gen-9")
        self.assertIn("Invalid workflow JSON", str(context.exception))

    def test_invalid_custom_values(self):
        handler = self.make_handler(
            {
                "workflow": json.dumps(CUSTOM_WORKFLOW),
                "custom-values": "[1, 2]",
            }
        )
        with self.assertRaises(RuntimeError) as context:
            handler.generate_image("a dog", "gen-10")
        self.assertIn("Custom Node Values", str(context.exception))

    # -- model discovery and connection test ------------------------------

    def test_fetch_models(self):
        handler = self.make_handler({"models": None})
        handler._values.pop("models", None)
        handler._fetch_models()
        self.assertEqual(
            handler.models,
            (
                ("modelA.safetensors", "modelA.safetensors"),
                ("modelB.safetensors", "modelB.safetensors"),
            ),
        )
        # Cached for the next start-up.
        self.assertIn("models", handler._values)

    def test_fetch_models_follows_model_node_class(self):
        # The workflow's model node is a CheckpointLoaderSimple; the stub
        # only lists models for that class, so the list is found.
        handler = self.make_handler({"workflow": json.dumps(CUSTOM_WORKFLOW)})
        handler._values.pop("models", None)
        handler._fetch_models()
        self.assertEqual(handler.models[0][0], "modelA.safetensors")

    def test_check_connection_success(self):
        handler = self.make_handler()
        handler._check_connection()
        messages = [message for message, _ in handler._thrown]
        self.assertTrue(any("Successfully connected" in m for m in messages))

    def test_check_connection_failure(self):
        handler = self.make_handler({"url": f"http://127.0.0.1:{free_port()}"})
        handler._check_connection()
        messages = [message for message, _ in handler._thrown]
        self.assertTrue(any("Could not connect" in m for m in messages))

    # -- failure modes ----------------------------------------------------

    def test_unreachable_server(self):
        handler = self.make_handler({"url": f"http://127.0.0.1:{free_port()}"})
        with self.assertRaises(RuntimeError) as context:
            handler.generate_image("a cat", "gen-11")
        self.assertIn("Could not connect", str(context.exception))

    def test_execution_error_reported(self):
        server = MockComfyUI(fail=True)
        try:
            handler = self.make_handler({"url": server.url})
            with self.assertRaises(RuntimeError) as context:
                handler.generate_image("a cat", "gen-12")
            self.assertIn("OSError", str(context.exception))
            self.assertIn("boom", str(context.exception))
        finally:
            server.stop()

    def test_timeout(self):
        server = MockComfyUI(delay=5.0)
        try:
            handler = self.make_handler({"url": server.url, "timeout": 2})
            with self.assertRaises(RuntimeError) as context:
                handler.generate_image("a cat", "gen-13")
            self.assertIn("Timed out", str(context.exception))
        finally:
            server.stop()


def main():
    unittest.main(module=__name__, argv=[sys.argv[0], "-v"], exit=True)


if __name__ == "__main__":
    main()