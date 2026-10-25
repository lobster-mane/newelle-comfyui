"""ComfyUI image generation extension for Newelle.

Adds an image generator backed by a ComfyUI server (local or remote)
through ComfyUI's HTTP API. The configurable options mirror the ComfyUI
integration of Open-WebUI: server address, optional API key, an exported
API-format workflow, node mapping, model discovery through /object_info,
image size, steps, batch size, negative prompt and seed.

Install: Newelle Settings -> Extensions -> import this file, then pick
"ComfyUI" as the image generation provider.
"""

import copy
import json
import os
import random
import threading
import time
import uuid
from gettext import gettext as _

import requests
from gi.repository import GLib

from .extensions import NewelleExtension
from .handlers import (
    ErrorSeverity,
    ExtraSettings,
    HandlerDescription,
    ImageGeneratorHandler,
)

# Server the extension talks to by default.
DEFAULT_URL = "http://127.0.0.1:8188"

# Seconds between /history/{prompt_id} polls while waiting for a generation.
POLL_INTERVAL = 1.5

# Per-request HTTP timeout.
HTTP_TIMEOUT = 30

# Upper bound for seeds, same range Open-WebUI uses for ComfyUI.
MAX_SEED = 1125899906842624

# Workflow node class types whose output images are collected after
# generation, same list used by Open-WebUI.
OUTPUT_NODE_TYPES = ("SaveImage", "SaveImageAdvanced", "PreviewImage")

# Basic text-to-image workflow in ComfyUI's API format, used when the user
# does not paste an exported one. Node layout matches ComfyUI's classic
# text-to-image example so the Node Mapping docs stay simple.
DEFAULT_WORKFLOW = {
    "3": {
        "class_type": "KSampler",
        "inputs": {
            "seed": 42,
            "steps": 20,
            "cfg": 8.0,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1.0,
            "model": ["4", 0],
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": ["5", 0],
        },
    },
    "4": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "v1-5-pruned-emaonly-fp16.safetensors"},
    },
    "5": {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": 512, "height": 512, "batch_size": 1},
    },
    "6": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "", "clip": ["4", 1]},
    },
    "7": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "", "clip": ["4", 1]},
    },
    "8": {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
    },
    "9": {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": "newelle", "images": ["8", 0]},
    },
}

# Preferred input key of each role, in order. The first key that exists in
# the node's inputs is used. Prompt and negative are resolved separately
# through _text_input_keys() because text encoders vary widely.
ROLE_INPUT_KEYS = {
    "width": ["width"],
    "height": ["height"],
    "steps": ["steps"],
    "seed": ["seed", "noise_seed"],
    "batch": ["batch_size"],
    "model": [],  # resolved through the *_name convention
}

# Additional input names of multi-CLIP text encoders (SD3, Flux, SDXL,
# HunyuanDiT, PixArt...) that receive the prompt text. Collected from
# ComfyUI's core nodes and common custom encoders.
KNOWN_TEXT_KEYS = (
    "clip_l",
    "clip_g",
    "t5xxl",
    "llama",
    "bert",
    "mt5xl",
    "text_l",
    "text_g",
    "qwen25_7b",
)


class ComfyUIHandler(ImageGeneratorHandler):
    """Image generator that runs workflows on a ComfyUI server.

    Only ``generate_image`` is implemented; the base class provides the
    ``generate_image`` LLM tool, the image mini app, caching and the chat
    widget handling.
    """

    key = "comfyui"

    def __init__(self, settings, path):
        super().__init__(settings, path)
        # Outcome of the last generate_image call per msg_uuid. Newelle
        # 1.5's generate_and_display has no error callback (errors are
        # only printed and on_done_callback fires regardless), so the
        # tool reads the outcome from here to report success or failure.
        self._generation_errors = {}
        self._generation_error_lock = threading.Lock()
        self.models = ()
        cached = self.get_setting("models", False)
        if cached:
            try:
                pairs = json.loads(cached)
                if isinstance(pairs, list) and pairs:
                    self.models = tuple(
                        (str(pair[0]), str(pair[1]))
                        for pair in pairs
                        if isinstance(pair, (list, tuple)) and len(pair) >= 2
                    )
            except (ValueError, TypeError):
                pass
        else:
            # First run: try to fill the model list in the background.
            threading.Thread(target=self._fetch_models, daemon=True).start()

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def get_extra_settings(self) -> list:
        custom_model = bool(self.get_setting("custom-model", False, False))
        if custom_model:
            model_setting = ExtraSettings.EntrySetting(
                "model",
                _("Model"),
                _(
                    "Name of the checkpoint or model file to use. "
                    "Leave empty to keep the model set in the workflow."
                ),
                "",
                website="https://www.comfy.org/",
            )
        else:
            model_setting = ExtraSettings.ComboSetting(
                "model",
                _("Model"),
                _(
                    "Model used for image generation. Keep the first "
                    "entry to use the model set in the workflow."
                ),
                (("", _("Use the workflow's model")),) + tuple(self.models),
                "",
                refresh=lambda x: self._refresh_models(),
            )

        random_seed = bool(self.get_setting("random-seed", False, True))
        seed_settings = []
        if not random_seed:
            seed_settings.append(
                ExtraSettings.SpinSetting(
                    "seed",
                    _("Fixed Seed"),
                    _("Seed used when Random Seed is disabled"),
                    0,
                    0,
                    MAX_SEED,
                    step=1,
                    page=100000,
                    round=0,
                )
            )

        return [
            ExtraSettings.EntrySetting(
                "url",
                _("Address"),
                _("ComfyUI server address, for example http://127.0.0.1:8188"),
                DEFAULT_URL,
                website="https://www.comfy.org/",
            ),
            ExtraSettings.EntrySetting(
                "api-key",
                _("API Key"),
                _(
                    "Bearer token for ComfyUI deployments that require "
                    "authentication. Leave empty for a local server."
                ),
                "",
                password=True,
            ),
            ExtraSettings.ButtonSetting(
                "check-connection",
                _("Test Connection"),
                _("Check that the ComfyUI server is reachable"),
                self._on_check_connection,
                _("Test"),
            ),
            ExtraSettings.ButtonSetting(
                "import-workflow",
                _("Import Workflow from File"),
                _(
                    "Load a workflow exported from ComfyUI in API format. "
                    "The file is validated on import."
                ),
                self._on_import_workflow,
                _("Import…"),
            ),
            ExtraSettings.MultilineEntrySetting(
                "workflow",
                _("Workflow (API Format)"),
                _(
                    "Paste a workflow exported from ComfyUI with "
                    "'Save (API Format)', or use the import button above. "
                    "Leave empty to use a basic default workflow."
                ),
                "",
                website="https://docs.comfy.org/api-reference/v2/overview",
            ),
            ExtraSettings.NestedSetting(
                "nodes",
                _("Node Mapping"),
                _(
                    "Which workflow nodes and inputs receive the prompt, "
                    "size, steps, seed and model. All fields are optional: "
                    "matching nodes and inputs are detected automatically "
                    "in most workflows."
                ),
                [
                    ExtraSettings.EntrySetting(
                        "model-node",
                        _("Model Node ID"),
                        _("Node that loads the checkpoint, e.g. 4"),
                        "",
                    ),
                    ExtraSettings.EntrySetting(
                        "model-key",
                        _("Model Input Key"),
                        _(
                            "Optional: name of the input that receives the "
                            "model, e.g. ckpt_name or unet_name. Leave "
                            "empty for automatic."
                        ),
                        "",
                    ),
                    ExtraSettings.EntrySetting(
                        "prompt-node",
                        _("Positive Prompt Node ID"),
                        _("Node that receives the prompt, e.g. 6"),
                        "",
                    ),
                    ExtraSettings.EntrySetting(
                        "prompt-key",
                        _("Positive Prompt Input Key"),
                        _(
                            "Optional: name of the input that receives the "
                            "prompt, e.g. text or prompt. Leave empty for "
                            "automatic."
                        ),
                        "",
                    ),
                    ExtraSettings.EntrySetting(
                        "negative-node",
                        _("Negative Prompt Node ID"),
                        _("Node that receives the negative prompt, e.g. 7"),
                        "",
                    ),
                    ExtraSettings.EntrySetting(
                        "negative-key",
                        _("Negative Prompt Input Key"),
                        _(
                            "Optional: name of the input that receives the "
                            "negative prompt, e.g. text or negative_prompt. "
                            "Leave empty for automatic."
                        ),
                        "",
                    ),
                    ExtraSettings.EntrySetting(
                        "width-node",
                        _("Width Node ID"),
                        _("Node that receives the image width, e.g. 5"),
                        "",
                    ),
                    ExtraSettings.EntrySetting(
                        "width-key",
                        _("Width Input Key"),
                        _("Optional: name of the input that receives the width. Leave empty for automatic."),
                        "",
                    ),
                    ExtraSettings.EntrySetting(
                        "height-node",
                        _("Height Node ID"),
                        _("Node that receives the image height, e.g. 5"),
                        "",
                    ),
                    ExtraSettings.EntrySetting(
                        "height-key",
                        _("Height Input Key"),
                        _("Optional: name of the input that receives the height. Leave empty for automatic."),
                        "",
                    ),
                    ExtraSettings.EntrySetting(
                        "steps-node",
                        _("Steps Node ID"),
                        _("Node that receives the step count, e.g. 3"),
                        "",
                    ),
                    ExtraSettings.EntrySetting(
                        "steps-key",
                        _("Steps Input Key"),
                        _("Optional: name of the input that receives the step count. Leave empty for automatic."),
                        "",
                    ),
                    ExtraSettings.EntrySetting(
                        "seed-node",
                        _("Seed Node ID"),
                        _("Node that receives the seed, e.g. 3"),
                        "",
                    ),
                    ExtraSettings.EntrySetting(
                        "seed-key",
                        _("Seed Input Key"),
                        _(
                            "Optional: name of the input that receives the "
                            "seed, e.g. seed or noise_seed. Leave empty "
                            "for automatic."
                        ),
                        "",
                    ),
                    ExtraSettings.EntrySetting(
                        "batch-node",
                        _("Batch Size Node ID"),
                        _("Node that receives the batch size, e.g. 5"),
                        "",
                    ),
                    ExtraSettings.EntrySetting(
                        "batch-key",
                        _("Batch Size Input Key"),
                        _("Optional: name of the input that receives the batch size. Leave empty for automatic."),
                        "",
                    ),
                    ExtraSettings.MultilineEntrySetting(
                        "custom-values",
                        _("Custom Node Values (JSON)"),
                        _(
                            'Advanced: JSON object with arbitrary node '
                            'inputs, e.g. {"3": {"cfg": 4.5}}'
                        ),
                        "",
                    ),
                ],
            ),
            ExtraSettings.ToggleSetting(
                "custom-model",
                _("Custom Model"),
                _("Type the model name manually instead of choosing from the list"),
                False,
                update_settings=True,
            ),
            model_setting,
            ExtraSettings.SpinSetting(
                "width",
                _("Width"),
                _("Image width in pixels"),
                512,
                64,
                4096,
                step=8,
                round=0,
            ),
            ExtraSettings.SpinSetting(
                "height",
                _("Height"),
                _("Image height in pixels"),
                512,
                64,
                4096,
                step=8,
                round=0,
            ),
            ExtraSettings.SpinSetting(
                "steps",
                _("Steps"),
                _("Number of sampling steps"),
                20,
                1,
                150,
                step=1,
                round=0,
            ),
            ExtraSettings.SpinSetting(
                "batch-size",
                _("Batch Size"),
                _("Images generated per run"),
                1,
                1,
                16,
                step=1,
                round=0,
            ),
            ExtraSettings.MultilineEntrySetting(
                "negative-prompt",
                _("Negative Prompt"),
                _("What the image should avoid"),
                "",
            ),
            ExtraSettings.ToggleSetting(
                "random-seed",
                _("Random Seed"),
                _("Use a new random seed for every generation"),
                True,
                update_settings=True,
            ),
            *seed_settings,
            ExtraSettings.MultilineEntrySetting(
                "prompt-template",
                _("Prompt Template"),
                _("Template for the final prompt, [input] is replaced by the user prompt"),
                "[input]",
            ),
            ExtraSettings.SpinSetting(
                "timeout",
                _("Timeout"),
                _("Maximum seconds to wait for a generation to complete"),
                300,
                30,
                3600,
                step=10,
                round=0,
            ),
        ]

    # ------------------------------------------------------------------
    # Server helpers
    # ------------------------------------------------------------------

    def _base_url(self) -> str:
        url = str(self.get_setting("url") or DEFAULT_URL).strip().rstrip("/")
        if url and "://" not in url:
            url = "http://" + url
        return url or DEFAULT_URL

    def _headers(self) -> dict:
        api_key = str(self.get_setting("api-key") or "").strip()
        if api_key:
            return {"Authorization": "Bearer " + api_key}
        return {}

    def _request_error(self, error) -> str:
        url = self._base_url()
        if isinstance(error, requests.exceptions.ConnectionError):
            return _(
                "Could not connect to ComfyUI at {url}. Make sure the "
                "server is running and the address is correct."
            ).format(url=url)
        if isinstance(error, requests.exceptions.Timeout):
            return _("ComfyUI at {url} took too long to respond.").format(url=url)
        return _("Error contacting ComfyUI at {url}: {error}").format(
            url=url, error=error
        )

    def _toast(self, message: str, severity) -> None:
        """Report a message through Newelle's toast system from any thread."""
        GLib.idle_add(self.throw, message, severity)

    # ------------------------------------------------------------------
    # Connection check and model discovery
    # ------------------------------------------------------------------

    def _on_check_connection(self, button):
        # Button callbacks run on the GTK thread; do the request elsewhere.
        threading.Thread(target=self._check_connection, daemon=True).start()

    def _check_connection(self):
        try:
            response = requests.get(
                self._base_url() + "/object_info",
                headers=self._headers(),
                timeout=10,
            )
        except requests.RequestException as error:
            self._toast(self._request_error(error), ErrorSeverity.WARNING)
            return
        if response.status_code != 200:
            self._toast(
                _("ComfyUI responded with HTTP {code}. Check the address and the API key.").format(
                    code=response.status_code
                ),
                ErrorSeverity.WARNING,
            )
            return
        self._toast(_("Successfully connected to ComfyUI."), ErrorSeverity.WARNING)

    def _refresh_models(self):
        threading.Thread(
            target=self._fetch_models, kwargs={"notify": True}, daemon=True
        ).start()

    def _get_object_info(self) -> dict:
        try:
            response = requests.get(
                self._base_url() + "/object_info",
                headers=self._headers(),
                timeout=15,
            )
        except requests.RequestException as error:
            raise RuntimeError(self._request_error(error)) from error
        if response.status_code != 200:
            raise RuntimeError(
                _("ComfyUI responded with HTTP {code} when reading the node list.").format(
                    code=response.status_code
                )
            )
        try:
            return response.json()
        except ValueError as error:
            raise RuntimeError(
                _("ComfyUI returned an unexpected response: {text}").format(
                    text=response.text[:200]
                )
            ) from error

    def _fetch_models(self, notify: bool = False):
        """Fetch the checkpoint list from /object_info.

        Follows Open-WebUI's logic: use the class type of the workflow's
        model node and pick its required input ending in ``_name``; fall
        back to ``CheckpointLoaderSimple.ckpt_name``.
        """
        try:
            info = self._get_object_info()
        except RuntimeError as error:
            if notify:
                self._toast(str(error), ErrorSeverity.WARNING)
            else:
                print("Error fetching ComfyUI models:", error)
            return
        models = self._models_from_object_info(info)
        if not models:
            if notify:
                self._toast(
                    _("No models were found on the ComfyUI server."),
                    ErrorSeverity.WARNING,
                )
            return
        self.models = tuple(models)
        try:
            self.set_setting("models", json.dumps(list(self.models)))
            self.settings_update()
        except Exception as error:  # noqa: BLE001 - never crash a background thread
            print("Error saving ComfyUI models:", error)

    def _models_from_object_info(self, info: dict) -> list:
        if not isinstance(info, dict):
            return []
        class_type = "CheckpointLoaderSimple"
        try:
            workflow = self._load_workflow()
        except RuntimeError:
            workflow = copy.deepcopy(DEFAULT_WORKFLOW)
        detected = self._detect_role_nodes(workflow)
        model_ids = detected.get("model") or []
        if model_ids:
            node = workflow.get(model_ids[0]) or {}
            detected_type = str(node.get("class_type") or "")
            if detected_type:
                class_type = detected_type

        values = []
        node_info = info.get(class_type)
        if isinstance(node_info, dict):
            required = (node_info.get("input") or {}).get("required") or {}
            name_key = next((key for key in required if key.endswith("_name")), None)
            if name_key and isinstance(required[name_key], (list, tuple)):
                choices = required[name_key][0] if required[name_key] else []
                values = [value for value in (choices or []) if isinstance(value, str)]
        if not values:
            # Same fallback Open-WebUI uses.
            fallback = info.get("CheckpointLoaderSimple") or {}
            required = (fallback.get("input") or {}).get("required") or {}
            ckpt = required.get("ckpt_name")
            if isinstance(ckpt, (list, tuple)) and ckpt and isinstance(ckpt[0], (list, tuple)):
                values = [value for value in (ckpt[0] or []) if isinstance(value, str)]
        return [(value, value) for value in values]

    # ------------------------------------------------------------------
    # Workflow loading and node mapping
    # ------------------------------------------------------------------

    def _on_import_workflow(self, button):
        """Open a file chooser and load a workflow exported in API format."""
        try:
            from gi.repository import Gio, Gtk
        except ImportError:
            return
        dialog = Gtk.FileDialog()
        dialog.set_title(_("Import ComfyUI Workflow (API Format)"))
        dialog.set_accept_label(_("Import"))
        filters = Gio.ListStore.new(Gtk.FileFilter)
        json_filter = Gtk.FileFilter()
        json_filter.set_name(_("JSON files"))
        json_filter.add_mime_type("application/json")
        json_filter.add_pattern("*.json")
        json_filter.add_pattern("*.workflow")
        filters.append(json_filter)
        dialog.set_filters(filters)

        def on_response(dialog, result):
            try:
                file = dialog.open_finish(result)
            except Exception:  # noqa: BLE001 - dialog cancelled or failed to open
                return
            if file is None:
                return
            try:
                _success, contents, _etag = file.load_contents(None)
                text = contents.decode("utf-8")
            except Exception as error:  # noqa: BLE001 - report any read error
                self._toast(
                    _("Could not read the workflow file: {error}").format(error=error),
                    ErrorSeverity.WARNING,
                )
                return
            try:
                self._validate_workflow_text(text)
            except RuntimeError as error:
                self._toast(str(error), ErrorSeverity.WARNING)
                return
            self.set_setting("workflow", text)
            self.settings_update()
            self._toast(_("Workflow imported successfully."), ErrorSeverity.WARNING)

        parent = button.get_root() if button is not None else None
        dialog.open(parent, None, on_response)

    @staticmethod
    def _validate_workflow_dict(workflow):
        """Check that a parsed workflow is in ComfyUI API export format."""
        if not isinstance(workflow, dict) or not workflow:
            raise RuntimeError(
                _("The workflow must be a non-empty JSON object in ComfyUI API format.")
            )
        if "nodes" in workflow and "links" in workflow:
            raise RuntimeError(
                _(
                    "This file was exported in the default (UI) format. "
                    "In ComfyUI export it again with 'Save (API Format)' "
                    "(or 'Export (API)') and import that file."
                )
            )
        for node_id, node in workflow.items():
            if not isinstance(node, dict) or "class_type" not in node or "inputs" not in node:
                raise RuntimeError(
                    _(
                        "The workflow is not in ComfyUI API format: node "
                        "{node} has no 'class_type' and 'inputs' entries. "
                        "Export the workflow with 'Save (API Format)' and "
                        "try again."
                    ).format(node=node_id)
                )

    def _validate_workflow_text(self, text: str):
        """Parse and validate a workflow file before importing it."""
        try:
            workflow = json.loads(text)
        except ValueError as error:
            raise RuntimeError(
                _("The file is not valid JSON: {error}").format(error=error)
            ) from error
        self._validate_workflow_dict(workflow)

    def _load_workflow(self) -> dict:
        raw = str(self.get_setting("workflow") or "").strip()
        if not raw:
            return copy.deepcopy(DEFAULT_WORKFLOW)
        try:
            workflow = json.loads(raw)
        except ValueError as error:
            raise RuntimeError(
                _(
                    "Invalid workflow JSON: {error}. In ComfyUI enable the "
                    "developer options and export the workflow with "
                    "'Save (API Format)', then paste the file content here."
                ).format(error=error)
            ) from error
        self._validate_workflow_dict(workflow)
        return workflow

    @staticmethod
    def _parse_node_ids(raw) -> list:
        ids = []
        for part in str(raw or "").split(","):
            part = part.strip()
            if part and part not in ids:
                ids.append(part)
        return ids

    @staticmethod
    def _sorted_node_ids(ids) -> list:
        def sort_key(node_id):
            text = str(node_id)
            return (0, int(text)) if text.isdigit() else (1, text)

        return sorted(ids, key=sort_key)

    def _detect_role_nodes(self, workflow: dict) -> dict:
        """Guess which nodes play which role, like Open-WebUI's node
        mapping does, but automatically for common workflows."""
        samplers, encoders, latents, loaders = [], [], [], []
        for node_id, node in workflow.items():
            class_type = str((node or {}).get("class_type", "")).lower()
            if "ksampler" in class_type:
                samplers.append(node_id)
            if "textencode" in class_type:
                encoders.append(node_id)
            if "empty" in class_type and "latent" in class_type:
                latents.append(node_id)
            if "checkpointloader" in class_type or "unetloader" in class_type:
                loaders.append(node_id)

        roles = {}
        if loaders:
            roles["model"] = loaders[:1]
        if latents:
            roles["width"] = latents[:1]
            roles["height"] = latents[:1]
            roles["batch"] = latents[:1]
        if samplers:
            roles["steps"] = samplers[:1]
            roles["seed"] = samplers[:1]
            sampler_inputs = (workflow[samplers[0]] or {}).get("inputs") or {}
            positive = sampler_inputs.get("positive")
            negative = sampler_inputs.get("negative")
            if isinstance(positive, (list, tuple)) and positive and positive[0] in workflow:
                roles["prompt"] = [positive[0]]
            if isinstance(negative, (list, tuple)) and negative and negative[0] in workflow:
                roles["negative"] = [negative[0]]
        sorted_encoders = self._sorted_node_ids(encoders)
        if "prompt" not in roles and sorted_encoders:
            roles["prompt"] = sorted_encoders[:1]
        if "negative" not in roles and len(sorted_encoders) > 1:
            roles["negative"] = sorted_encoders[1:2]
        return roles

    def _role_nodes(self, workflow: dict, detected: dict, role: str):
        """Return ``(node_ids, manual)`` for a role, preferring the setting."""
        raw = self.get_setting(role + "-node")
        if raw:
            node_ids = self._parse_node_ids(raw)
            for node_id in node_ids:
                if node_id not in workflow:
                    raise RuntimeError(
                        _(
                            "Node {node} (set for {role}) was not found in "
                            "the workflow. Node IDs are the numbers shown "
                            "on each node in the ComfyUI editor."
                        ).format(node=node_id, role=role)
                    )
            return node_ids, True
        return detected.get(role) or [], False

    @staticmethod
    def _node_input_key(node: dict, role: str):
        inputs = (node or {}).get("inputs") or {}
        candidates = ROLE_INPUT_KEYS.get(role) or []
        for key in candidates:
            if key in inputs:
                return key
        return candidates[0] if candidates else None

    @staticmethod
    def _model_input_key(node: dict) -> str:
        inputs = (node or {}).get("inputs") or {}
        for key in inputs:
            if key.endswith("_name"):
                return key
        return "ckpt_name"

    @staticmethod
    def _text_input_keys(node: dict, role: str) -> list:
        """Prompt-like input keys of a text encoder for the given role.

        Core nodes use ``text``; custom encoders use ``prompt``,
        ``positive_prompt`` or ``negative_prompt``; multi-CLIP encoders
        expose one input per text model (``clip_l``, ``t5xxl``, ...).
        Some encoders (Qwen, Wan) combine positive and negative in one
        node, so the negative role prefers keys containing ``negative``
        and never reuses the positive keys of the same node.
        """
        inputs = (node or {}).get("inputs") or {}
        positive_like = [
            key
            for key in inputs
            if (key == "text" or "prompt" in key or key in KNOWN_TEXT_KEYS)
            and "negative" not in key
        ]
        negative_like = [key for key in inputs if "negative" in key]
        if role == "negative":
            return negative_like or positive_like
        return positive_like

    def _apply_role(self, workflow: dict, role: str, node_ids, value, manual: bool):
        override = str(self.get_setting(role + "-key") or "").strip()
        for node_id in node_ids:
            node = workflow.get(node_id) or {}
            if override:
                # The user picked the input key explicitly: apply it as is.
                node.setdefault("inputs", {})[override] = value
                continue
            if role in ("prompt", "negative"):
                keys = self._text_input_keys(node, role)
                if not keys:
                    if role == "prompt":
                        raise RuntimeError(
                            _(
                                "Node {node} has no text input for the "
                                "prompt (looked for text, prompt or "
                                "multi-CLIP inputs). Set the Positive "
                                "Prompt Input Key in the Node Mapping "
                                "settings to choose one."
                            ).format(node=node_id)
                        )
                    continue
            elif role == "model":
                keys = [self._model_input_key(node)]
            else:
                keys = [self._node_input_key(node, role)]
            inputs = node.setdefault("inputs", {})
            for key in keys:
                if not manual and key not in inputs:
                    # Auto-detected nodes only accept values for inputs
                    # they actually have; use Custom Node Values for
                    # anything else.
                    continue
                inputs[key] = value

    def _apply_custom_values(self, workflow: dict):
        raw = str(self.get_setting("custom-values") or "").strip()
        if not raw:
            return
        try:
            data = json.loads(raw)
        except ValueError as error:
            raise RuntimeError(
                _("Invalid Custom Node Values JSON: {error}").format(error=error)
            ) from error
        if not isinstance(data, dict):
            # RuntimeError keeps every user-facing failure on one code path.
            raise RuntimeError(  # noqa: TRY004 - intentional, see comment
                _(
                    'Custom Node Values must be a JSON object such as '
                    '{"3": {"cfg": 4.5}}'
                )
            )
        for node_id, pairs in data.items():
            node = workflow.get(str(node_id))
            if node is None:
                raise RuntimeError(
                    _(
                        "Node {node} from Custom Node Values was not found "
                        "in the workflow."
                    ).format(node=node_id)
                )
            if not isinstance(pairs, dict):
                raise RuntimeError(  # noqa: TRY004 - intentional, see comment
                    _(
                        'Custom Node Values must map node IDs to objects '
                        'such as {"3": {"cfg": 4.5}}'
                    )
                )
            node.setdefault("inputs", {}).update(pairs)

    def _build_workflow(self, prompt: str) -> dict:
        workflow = self._load_workflow()
        detected = self._detect_role_nodes(workflow)

        width = int(self.get_setting("width") or 512)
        height = int(self.get_setting("height") or 512)
        steps = int(self.get_setting("steps") or 20)
        batch = int(self.get_setting("batch-size") or 1)
        negative = str(self.get_setting("negative-prompt") or "").strip()
        model = str(self.get_setting("model") or "").strip()
        if bool(self.get_setting("random-seed")):
            seed = random.randint(0, MAX_SEED)
        else:
            seed = int(self.get_setting("seed") or 0)

        prompt_ids, prompt_manual = self._role_nodes(workflow, detected, "prompt")
        if not prompt_ids:
            raise RuntimeError(
                _(
                    "No text prompt node was found in the workflow. Make "
                    "sure the workflow was exported in API format, or set "
                    "the Positive Prompt Node ID in the Node Mapping "
                    "settings."
                )
            )
        self._apply_role(workflow, "prompt", prompt_ids, prompt, prompt_manual)

        if negative:
            node_ids, manual = self._role_nodes(workflow, detected, "negative")
            if node_ids:
                self._apply_role(workflow, "negative", node_ids, negative, manual)

        if model:
            node_ids, manual = self._role_nodes(workflow, detected, "model")
            if not node_ids:
                raise RuntimeError(
                    _(
                        "A model is selected but no checkpoint or UNET "
                        "loader node was found in the workflow. Set the "
                        "Model Node ID in the Node Mapping settings, or "
                        "keep 'Use the workflow's model' selected."
                    )
                )
            self._apply_role(workflow, "model", node_ids, model, manual)

        for role, value in (
            ("width", width),
            ("height", height),
            ("steps", steps),
            ("seed", seed),
            ("batch", batch),
        ):
            node_ids, manual = self._role_nodes(workflow, detected, role)
            if node_ids:
                self._apply_role(workflow, role, node_ids, value, manual)

        self._apply_custom_values(workflow)
        return workflow

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _format_queue_error(self, response) -> str:
        try:
            data = response.json()
        except ValueError:
            detail = (response.text or "").strip()[:300]
            return _("ComfyUI rejected the workflow (HTTP {code}): {detail}").format(
                code=response.status_code, detail=detail or _("no details")
            )
        parts = []
        error = data.get("error") or {}
        if error.get("message"):
            parts.append(str(error["message"]))
        if error.get("details"):
            parts.append(str(error["details"]))
        for node_id, node_error in (data.get("node_errors") or {}).items():
            for item in (node_error.get("errors") or []):
                detail = item.get("details") or item.get("message")
                if detail:
                    parts.append(
                        _("[node {node}] {detail}").format(
                            node=node_id, detail=detail
                        )
                    )
        unique = list(dict.fromkeys(str(part) for part in parts))
        return _("ComfyUI rejected the workflow (HTTP {code}): {detail}").format(
            code=response.status_code, detail="; ".join(unique) or _("unknown error")
        )

    def _queue_prompt(self, workflow: dict, client_id: str) -> str:
        try:
            response = requests.post(
                self._base_url() + "/prompt",
                json={"prompt": workflow, "client_id": client_id},
                headers=self._headers(),
                timeout=HTTP_TIMEOUT,
            )
        except requests.RequestException as error:
            raise RuntimeError(self._request_error(error)) from error
        if response.status_code != 200:
            raise RuntimeError(self._format_queue_error(response))
        try:
            prompt_id = response.json().get("prompt_id")
        except ValueError as error:
            raise RuntimeError(
                _("ComfyUI returned an unexpected response: {text}").format(
                    text=response.text[:200]
                )
            ) from error
        if not prompt_id:
            raise RuntimeError(_("ComfyUI did not return a prompt id."))
        return prompt_id

    @staticmethod
    def _check_history_status(entry: dict):
        status = entry.get("status") or {}
        if status.get("status_str") != "error":
            return
        details = []
        for message in status.get("messages") or []:
            if not isinstance(message, (list, tuple)) or len(message) < 2:
                continue
            kind = message[0]
            data = message[1] if isinstance(message[1], dict) else {}
            if kind == "execution_error":
                location = ""
                if data.get("node_id") or data.get("node_type"):
                    location = _(" (node {node}: {node_type})").format(
                        node=data.get("node_id"), node_type=data.get("node_type")
                    )
                exception = str(data.get("exception_type") or _("Error"))
                exception_message = str(data.get("exception_message") or "")
                details.append(
                    exception + location + (": " + exception_message if exception_message else "")
                )
            elif kind == "execution_interrupted":
                details.append(_("Execution was interrupted"))
        raise RuntimeError(
            _("ComfyUI failed to generate the image: {details}").format(
                details="; ".join(details) or _("execution error")
            )
        )

    def _prompt_lost(self, prompt_id: str) -> bool:
        """Check whether the prompt is still known to ComfyUI.

        A prompt that is in neither the queue nor the history was dropped:
        the server may have been restarted (the history is wiped) or the
        prompt was cancelled. Polling would otherwise wait for the whole
        timeout with nothing to wait for.
        """
        try:
            response = requests.get(
                self._base_url() + "/queue",
                headers=self._headers(),
                timeout=15,
            )
        except requests.RequestException:
            return False  # transient queue problems are not fatal
        if response.status_code != 200:
            return False
        try:
            data = response.json()
        except ValueError:
            return False
        for entry in list(data.get("queue_running") or []) + list(
            data.get("queue_pending") or []
        ):
            if (
                isinstance(entry, (list, tuple))
                and len(entry) > 1
                and entry[1] == prompt_id
            ):
                return False
        return True

    def _wait_for_history(self, prompt_id: str, timeout: int) -> dict:
        """Poll /history/{prompt_id} until the generation finishes."""
        url = self._base_url() + "/history/" + str(prompt_id)
        deadline = time.monotonic() + timeout
        missing_count = 0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    _(
                        "Timed out after {seconds} seconds waiting for "
                        "ComfyUI to finish generating. Increase the "
                        "Timeout setting or reduce the image size."
                    ).format(seconds=int(timeout))
                )
            try:
                response = requests.get(
                    url,
                    headers=self._headers(),
                    timeout=min(HTTP_TIMEOUT, max(5.0, remaining)),
                )
            except requests.RequestException as error:
                raise RuntimeError(self._request_error(error)) from error
            if response.status_code != 200:
                raise RuntimeError(
                    _(
                        "ComfyUI responded with HTTP {code} while checking "
                        "the generation status."
                    ).format(code=response.status_code)
                )
            try:
                data = response.json()
            except ValueError:
                data = None
            entry = data.get(prompt_id) if isinstance(data, dict) else None
            if entry is not None:
                self._check_history_status(entry)
                return entry
            # Not finished yet: make sure ComfyUI still knows the prompt.
            # A few consecutive misses are tolerated to avoid races with
            # the moment a prompt moves from the queue into the history.
            if self._prompt_lost(prompt_id):
                missing_count += 1
                if missing_count >= 3:
                    raise RuntimeError(
                        _(
                            "The generation disappeared from the ComfyUI "
                            "queue. The server may have been restarted or "
                            "crashed, or the prompt was cancelled."
                        )
                    )
            else:
                missing_count = 0
            time.sleep(min(POLL_INTERVAL, max(0.05, deadline - time.monotonic())))

    @staticmethod
    def _collect_images(workflow: dict, entry: dict) -> list:
        outputs = entry.get("outputs") or {}
        images = []
        for node_id, node in workflow.items():
            if (node or {}).get("class_type") not in OUTPUT_NODE_TYPES:
                continue
            node_output = outputs.get(node_id) or {}
            for image in node_output.get("images") or []:
                if isinstance(image, dict) and image.get("filename"):
                    images.append(image)
        if not images:
            # Fallback for custom save nodes: take every image output.
            for node_output in outputs.values():
                for image in (node_output or {}).get("images") or []:
                    if isinstance(image, dict) and image.get("filename"):
                        images.append(image)
        return images

    def _download_image(self, image: dict, path: str) -> str:
        params = {
            "filename": image.get("filename", ""),
            "subfolder": image.get("subfolder", ""),
            "type": image.get("type", "output"),
        }
        try:
            response = requests.get(
                self._base_url() + "/view",
                params=params,
                headers=self._headers(),
                timeout=HTTP_TIMEOUT * 2,
            )
            response.raise_for_status()
        except requests.RequestException as error:
            raise RuntimeError(self._request_error(error)) from error
        data = response.content
        try:
            from io import BytesIO

            from PIL import Image

            picture = Image.open(BytesIO(data))
            if picture.mode not in ("RGB", "RGBA"):
                picture = picture.convert("RGBA")
            picture.save(path, "PNG")
        except Exception:  # noqa: BLE001 - fallback: keep the original bytes
            # Not a PIL-readable image: keep the original bytes.
            with open(path, "wb") as file:
                file.write(data)
        return path

    def generate_image(self, prompt: str, msg_uuid: str, output_file: str | None = None) -> str:
        """Generate an image with ComfyUI and return the local file path.

        Runs on a worker thread through the base class, so no GTK calls
        happen here. The outcome is tracked per ``msg_uuid`` so the
        ``generate_image`` tool can report success or failure.
        """
        try:
            path = self._generate_image_impl(prompt, msg_uuid, output_file)
        except Exception as error:
            with self._generation_error_lock:
                self._generation_errors[msg_uuid] = str(error)
            raise
        with self._generation_error_lock:
            self._generation_errors.pop(msg_uuid, None)
        return path

    def _generate_image_impl(self, prompt: str, msg_uuid: str, output_file: str | None) -> str:
        if output_file is None:
            output_file = self.cache_path_for(msg_uuid)

        template = str(self.get_setting("prompt-template") or "")
        if template and "[input]" in template:
            prompt = template.replace("[input]", prompt)

        workflow = self._build_workflow(prompt)
        client_id = uuid.uuid4().hex
        prompt_id = self._queue_prompt(workflow, client_id)
        timeout = int(self.get_setting("timeout") or 300)
        entry = self._wait_for_history(prompt_id, timeout)
        images = self._collect_images(workflow, entry)
        if not images:
            raise RuntimeError(
                _(
                    "ComfyUI finished but did not produce any image. Check "
                    "that the workflow contains a SaveImage node."
                )
            )
        for index, image in enumerate(images):
            target = output_file if index == 0 else os.path.join(
                self.cache_dir, f"{msg_uuid}-{index}.png"
            )
            self._download_image(image, target)
        return output_file

    # ------------------------------------------------------------------
    # Tool
    # ------------------------------------------------------------------

    def _generate_image_tool(self, prompt: str, msg_uuid=None):
        """Like the base class tool, but the result always carries an output.

        Newelle 1.5's default image tool never releases its ToolResult,
        so the generation chain blocks forever and the chat's stop button
        stays after the image is done. This tool releases the result
        with a short textual report instead, which makes the assistant
        close the turn through the same path every other tool uses; the
        image itself is shown through the widget.

        The base class worker only provides ``on_done_callback`` (it
        fires on success and failure alike; errors are just printed), so
        the outcome recorded by ``generate_image`` decides the report.
        """
        from .tools import ToolResult
        from .ui.widgets.image_generator import ImageGeneratorWidget

        widget = ImageGeneratorWidget(width=400, height=400)
        widget.set_prompt(prompt)
        result = ToolResult()
        result.set_widget(widget)

        def on_done():
            with self._generation_error_lock:
                error = self._generation_errors.pop(msg_uuid, None)
            if error is not None:
                result.set_output(
                    _("Image generation failed: {message}").format(message=error)
                )
            else:
                result.set_output(
                    _("The image was generated successfully and is displayed above.")
                )

        self.generate_and_display(prompt, widget, msg_uuid, on_done_callback=on_done)
        return result

    def get_tools(self) -> list:
        tools = super().get_tools()
        # The base class builds the Tool with self._generate_image_tool,
        # which already resolves to the override above through normal
        # attribute lookup. Swap it explicitly anyway so the tool keeps
        # carrying a result if a future base version binds differently.
        for tool in tools:
            if tool.name == "generate_image":
                tool.func = self._generate_image_tool
        return tools


class ComfyUIExtension(NewelleExtension):
    """Registers the ComfyUI image generator with Newelle."""

    id = "comfyui"
    name = "ComfyUI"

    def get_image_generator_handlers(self) -> list:
        return [
            HandlerDescription(
                key=ComfyUIHandler.key,
                title=_("ComfyUI"),
                description=_(
                    "Generate images with a local or remote ComfyUI server"
                ),
                handler_class=ComfyUIHandler,
                website="https://www.comfy.org/",
            )
        ]