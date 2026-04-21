# Spark LTX

A minimal batch driver for running **LTX-2.3 FLF2V** workflows against a local ComfyUI server, with one anti-hallucination trick: feed the **same keyframe as both first and last frame**, anchor the first at full strength, and anchor the last softly. This pins the scene to the keyframe without collapsing motion.

## Why

Pure I2V on LTX-2.3 drifts. Subjects morph, the camera wanders, and over a few seconds the scene can walk off the keyframe. Pure FLF2V with two different end frames forces the model to morph toward the target, often producing weird intermediate states (a lighthouse growing into a windmill, a face becoming someone else).

Feeding the **same image** at both ends with asymmetric strengths — `first=1.0`, `last=0.2..0.5` — tethers the model to the keyframe the entire clip while allowing the middle to move. No target to morph toward, no drift.

## Requirements

- Python 3.10+
- A running ComfyUI server with the LTX-2.3 nodes installed, plus the 22B distilled FP8 checkpoint and Gemma-3 12B FP4 text encoder. See [Lightricks/ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo).

## Install

```bash
git clone https://github.com/netmana808/spark-ltx.git
cd spark-ltx
pip install -e .
```

This exposes `spark-ltx-flf2v` on your `PATH`.

## Use

Prepare a story directory:

```
my_story/
├── plan.json
└── keyframes/
    ├── 01.png
    ├── 02.png
    └── 03.png
```

Write `plan.json` (see [examples/plan.example.json](examples/plan.example.json)):

```json
{
  "title": "My Story",
  "scenes": [
    {
      "id": "01",
      "duration_s": 3,
      "prompt": "slow cinematic push-in, gentle breeze",
      "last_strength": 0.4
    },
    {
      "id": "02",
      "duration_s": 3,
      "prompt": "camera locked, subject breathes",
      "last_strength": 0.5
    }
  ]
}
```

Run:

```bash
spark-ltx-flf2v my_story/plan.json workflows/ltx23_flf2v.json \
  --width 1024 --height 1536 --fps 24 \
  --comfy-url http://localhost:8188 \
  --comfy-input-dir /path/to/ComfyUI/input
```

Clips land in `my_story/clips/<id>.mp4`. Existing non-empty clips are skipped on re-run.

## Configuration

| Flag | Env var | Default | Notes |
|---|---|---|---|
| `--comfy-url` | `COMFY_URL` | `http://localhost:8188` | ComfyUI REST base URL |
| `--comfy-input-dir` | `COMFY_INPUT_DIR` | `./input` | Where keyframes are staged for the server |
| `--fps` | — | `24` | |
| `--width` | — | `1280` | Must be divisible by 32 |
| `--height` | — | `704` | Must be divisible by 32 |
| `--first-strength` | — | `1.0` | First-frame anchor strength |
| `--last-strength` | — | `0.3` | Last-frame anchor strength; lower = more mid-clip motion |
| `--scene` | — | — | Run a single scene by id |

Per-scene `first_strength` and `last_strength` in `plan.json` override the CLI defaults.

## Tuning `last_strength`

| Value | Use for |
|---|---|
| `0.0` | Scenes where the subject traverses the frame (no snap-back desired). |
| `0.2` | Explosions, camera whips, kinetic beats. |
| `0.3 – 0.4` | Default. Normal dolly moves. |
| `0.5` | Static or near-static scenes, breathing loops, clean returns. |
| `1.0` | Aggressive anchoring; reduces motion significantly. |

## Plan schema

| Field | Type | Notes |
|---|---|---|
| `title` | string | Display title. |
| `scenes[].id` | string | Matches `keyframes/<id>.png`. |
| `scenes[].duration_s` | int | Seconds. |
| `scenes[].prompt` | string | Motion / camera / mood description. |
| `scenes[].first_strength` | float | Optional override. |
| `scenes[].last_strength` | float | Optional override. |

## Workflow compatibility

The runner matches ComfyUI nodes by `class_type` and `_meta.title`, not numeric IDs, so it tolerates workflow edits and re-saves. It expects:

- Two `LoadImage` nodes (first + last frame) — both get the same image.
- Two `LTXVAddGuide` nodes with `frame_idx` `0` and `-1` — these receive the configured strengths.
- `PrimitiveInt` nodes titled `Width`, `Height` / `height`, `Duration`, `Frame Rate` / `Frame Rate(int)`.
- A `CLIPTextEncode` node titled `CLIP Text Encode (Prompt)` with a prompt that does **not** start with `blurry` or `out of` (that's the convention used to distinguish the positive prompt from the negative).

A sanitized starting-point workflow is provided at [workflows/ltx23_flf2v.json](workflows/ltx23_flf2v.json).

## License

MIT.
