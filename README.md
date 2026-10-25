# Newelle ComfyUI

[![Newelle extension](https://img.shields.io/badge/Newelle-Extension-0068d6)](https://github.com/topics/newelle-extension)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/License-GPL--3.0--or--later-blue.svg)](LICENSE)

A [Newelle](https://github.com/qwersyk/Newelle) extension that adds
**ComfyUI** as an image generation provider, so Newelle's built-in
`generate_image` tool and Image mini app can run ComfyUI workflows on a
local or remote server.

The available options mirror the [ComfyUI integration of
Open-WebUI](https://github.com/open-webui/open-webui): server address,
optional API key, an exported API-format workflow, node mapping, model
discovery through `/object_info`, image size, steps, batch size, negative
prompt and seed.

## Installation

1. Download [`comfyui.py`](comfyui.py) from this repository.
2. In Newelle, open **Settings → Extensions** and import the file (or drop
   it onto the Extensions page).
3. Open **Settings → Knowledge → Image Generation** and select **ComfyUI** as
   the provider.
4. If ComfyUI runs on the same machine with default settings, everything
   already works: a basic text-to-image workflow is built in. Press
   **Test** to verify the connection, and the refresh button next to
   **Model** to list the checkpoints available on your server.

> **Note:** Newelle's `generate_image` tool is disabled by default. Enable
> it in **Settings → Tools** if you want the LLM to generate images, or
> just use the Image mini app (the "Images" tab entry) which uses the
> same provider.

## Using your own workflows

The default workflow is a simple SD-style text-to-image graph. To use a
real pipeline (Flux, SDXL, LoRAs, upscalers, ...), export it from
ComfyUI:

1. In ComfyUI, enable the developer options (*Settings → App → See "Dev
   mode"* in classic UIs, or the equivalent toggle in current versions).
2. Load your workflow and export it in **API format** ("Save (API
   Format)" / "Export (API)").
3. Press **Import…** next to the **Workflow (API Format)** setting and pick
   the exported file, or paste its JSON into the setting. The import
   validates the format and tells you if you picked the wrong export (the
   default *UI* format) instead of an API export.

The model selection is optional: by default the combo stays on
**Use the workflow's model**, so whatever checkpoint the exported
workflow references is left untouched.

The extension then rewrites the relevant nodes on every generation:

| Setting | Written into the workflow |
| --- | --- |
| Model | the checkpoint/UNET loader input (e.g. `ckpt_name`, `unet_name`) — only when a model is selected |
| Prompt Template + user prompt | the positive text encoder |
| Negative Prompt | the negative text encoder |
| Width / Height | the empty latent image node |
| Steps / Seed | the sampler (`seed` or `noise_seed` for `KSamplerAdvanced`) |
| Batch Size | the empty latent image node (`batch_size`) |

### Node mapping

Roles are detected automatically for common graphs: checkpoint/UNET
loaders for the model, the sampler's `positive`/`negative` links for the
text encoders, empty-latent nodes for size and batch size, and `KSampler`
nodes for steps and seed. If detection picks the wrong node (or your
graph uses several), set the node IDs manually in the **Node Mapping**
section: each field accepts one or more comma-separated IDs, e.g. `6` or
`4, 5`. Every role also has an optional **Input Key** field to choose the
exact input name that receives the value (e.g. `noise_seed` for a
`KSamplerAdvanced`, or a custom node input); when set, it is applied as
is and takes precedence over automatic detection.

Input names are resolved per node: the prompt is written to `text`,
`prompt`, `positive_prompt`, `user_prompt` or the multi-CLIP inputs
(`clip_l`, `clip_g`, `t5xxl`, ...) depending on the encoder; the negative
prompt goes to the keys containing `negative` (combined Qwen/Wan
encoders keep both on a single node and are handled correctly). Models
land in the loader's `*_name` input (`ckpt_name`, `unet_name`, ...), and
seeds go to `seed` or `noise_seed` (`KSamplerAdvanced`).

For anything else there is **Custom Node Values (JSON)**, applied last
and overriding everything else:

```json
{"3": {"cfg": 4.5}, "12": {"text": "extra conditioning"}}
```

## Settings reference

| Setting | Default | Description |
| --- | --- | --- |
| Address | `http://127.0.0.1:8188` | ComfyUI server URL. |
| API Key | *(empty)* | Bearer token for deployments that require authentication. |
| Test Connection | — | Checks that the server answers on `/object_info`. |
| Import Workflow | — | Loads a workflow file and validates that it is an API export. |
| Workflow (API Format) | *(empty = built-in default)* | Workflow JSON exported as API format. |
| Node Mapping | *(auto)* | Per-role node IDs, optional per-role input keys, and custom node values. |
| Custom Model | off | Type a model name instead of choosing from the list. |
| Model | Use the workflow's model | Optional override; refresh the list from `/object_info`. |
| Width / Height | 512 / 512 | Image size in pixels. |
| Steps | 20 | Sampling steps. |
| Batch Size | 1 | Images per run; extra outputs are cached next to the displayed one. |
| Negative Prompt | *(empty)* | Injected only when not empty. |
| Random Seed | on | New seed for every generation (range like Open-WebUI, up to 2⁵⁰). |
| Prompt Template | `[input]` | Newelle convention: `[input]` is replaced by the user prompt. |
| Timeout | 300 s | Maximum wait for one generation. |

## How it works

The extension speaks ComfyUI's HTTP API directly (no extra Python
dependencies — `requests` and Pillow ship with Newelle):

1. `POST /prompt` with the patched workflow and a fresh client id.
2. Poll `GET /history/{prompt_id}` until the execution finishes, while
   `GET /queue` confirms the prompt is still known to the server. ComfyUI
   execution errors are surfaced with their node and exception details,
   and a prompt that vanishes from both the queue and the history (for
   example after a server crash or restart) fails immediately instead
   of waiting for the timeout.
3. Collect the images produced by `SaveImage`, `SaveImageAdvanced` and
   `PreviewImage` nodes from the history entry.
4. Download each image through `GET /view` and save it as PNG into
   Newelle's generated-images cache, where the chat widget picks it up.

Model discovery uses `GET /object_info`: the class type of the mapped
model node decides which input list is offered (any required input ending
in `_name`, e.g. `ckpt_name` or `unet_name`), falling back to
`CheckpointLoaderSimple.ckpt_name` — the same approach Open-WebUI uses.

## Development

The development environment is NixOS; enter the shell with:

```console
$ nix-shell
newelle-comfyui development shell

  Lint:  ruff check comfyui.py tests/
  Test:  python tests/mock_comfyui.py
```

`tests/mock_comfyui.py` runs the whole extension headlessly: it stubs the
Newelle modules exactly like the extension loader does, starts a fake
ComfyUI server (`/object_info`, `/prompt`, `/history`, `/queue`, `/view`)
and checks the default workflow, auto-detection, node and input-key
overrides, workflow import validation, optional model selection, model
discovery, error propagation, lost prompts and timeouts.

## Troubleshooting

- **"Could not connect to ComfyUI"** — is the server running? Default
  address is `http://127.0.0.1:8188` (start ComfyUI with `python main.py`
  and check the port it prints). If you changed the port, update the
  Address setting.
- **"This file was exported in the default (UI) format"** — you exported
  or pasted the *UI* export instead of the *API* export. Enable the
  developer options in ComfyUI and use "Save (API Format)".
- **"The generation disappeared from the ComfyUI queue"** — ComfyUI was
  restarted or crashed while generating (its history is wiped on
  restart). Try again; if your models keep crashing the server, check
  the ComfyUI logs.
- **Model list is empty or stale** — press the refresh button next to the
  Model setting while the server is reachable.
- **"checkpoint not found"** during generation — the selected model is
  not on the server; refresh the model list, or switch the Model setting
  back to *Use the workflow's model*.
- **Generation takes very long** — raise the Timeout setting; big images
  and many steps on CPU can exceed 300 seconds.

## Credits

- The option set and API flow follow Open-WebUI's ComfyUI image
  generation integration (`open-webui/routers/images.py` and
  `open-webui/utils/images/comfyui.py`).
- Handler and extension structure follow the
  [Newelle extension documentation](https://github.com/qwersyk/Newelle/wiki/Developing-extensions).

## License

[GPL-3.0-or-later](LICENSE) — the same license as Newelle itself.