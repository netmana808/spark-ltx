"""Batch driver for LTX-2.3 FLF2V with weighted same-keyframe anchoring.

The runner feeds the same keyframe as both first and last frame of a FLF2V
workflow. The first-frame guide is anchored at ``first_strength`` (default
1.0); the last-frame guide is anchored at ``last_strength`` (default 0.3).
Asymmetric anchoring pins the scene to the keyframe while allowing camera and
lighting motion through the middle of the clip.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from .comfy import ComfyClient

DEFAULT_COMFY_URL = os.environ.get("COMFY_URL", "http://localhost:8188")
DEFAULT_COMFY_INPUT_DIR = Path(os.environ.get("COMFY_INPUT_DIR", "./input"))
MIN_CLIP_BYTES = 10_000


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def patch_workflow(
    workflow: dict[str, Any],
    *,
    image_name: str,
    prompt: str,
    width: int,
    height: int,
    duration: int,
    fps: int,
    first_strength: float,
    last_strength: float,
) -> None:
    """Patch an LTX-2.3 FLF2V workflow in place.

    Nodes are matched by ``class_type`` and ``_meta.title`` so the patcher
    tolerates edits and re-saves of the template.
    """
    for node in workflow.values():
        class_type = node.get("class_type", "")
        title = node.get("_meta", {}).get("title", "")
        inputs = node.setdefault("inputs", {})

        if class_type == "LoadImage":
            inputs["image"] = image_name
        elif class_type == "CLIPTextEncode" and title == "CLIP Text Encode (Prompt)":
            existing = inputs.get("text", "")
            # Distinguish the positive prompt from the negative by its opening
            # word. Negative prompts in LTX workflows conventionally start with
            # "blurry" / "out of focus".
            if isinstance(existing, str) and not existing.lower().startswith(
                ("blurry", "out of")
            ):
                inputs["text"] = prompt
        elif class_type == "PrimitiveInt":
            if title == "Width":
                inputs["value"] = width
            elif title in ("Height", "height"):
                inputs["value"] = height
            elif title == "Duration":
                inputs["value"] = duration
            elif title in ("Frame Rate", "Frame Rate(int)"):
                inputs["value"] = fps
        elif class_type == "LTXVAddGuide":
            frame_idx = inputs.get("frame_idx")
            if frame_idx == 0:
                inputs["strength"] = first_strength
            elif frame_idx == -1:
                inputs["strength"] = last_strength


def run_scene(
    client: ComfyClient,
    workflow_template: dict[str, Any],
    *,
    keyframe: Path,
    output: Path,
    comfy_input_dir: Path,
    scene_id: str,
    story_id: str,
    prompt: str,
    width: int,
    height: int,
    duration: int,
    fps: int,
    first_strength: float,
    last_strength: float,
) -> bool:
    staged_name = f"{story_id}_flf_{scene_id}.png"
    comfy_input_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(keyframe, comfy_input_dir / staged_name)

    workflow = deepcopy(workflow_template)
    patch_workflow(
        workflow,
        image_name=staged_name,
        prompt=prompt,
        width=width,
        height=height,
        duration=duration,
        fps=fps,
        first_strength=first_strength,
        last_strength=last_strength,
    )

    log(f"  {scene_id}: submit first={first_strength} last={last_strength} prompt={prompt[:60]!r}")
    started = time.time()
    prompt_id = client.submit(workflow)
    try:
        result = client.poll(prompt_id)
    except (TimeoutError, RuntimeError) as exc:
        log(f"  {scene_id}: FAILED — {exc}")
        return False

    ok, size = client.download(result.get("outputs", {}), output)
    elapsed = time.time() - started
    if ok:
        log(f"  {scene_id}: OK {elapsed:.0f}s {size / 1024:.0f} KB")
    else:
        log(f"  {scene_id}: no output after {elapsed:.0f}s")
    return ok


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run LTX-2.3 FLF2V with weighted same-keyframe anchoring "
            "across every scene in a plan.json."
        ),
    )
    parser.add_argument("plan", type=Path, help="Path to plan.json")
    parser.add_argument(
        "workflow",
        type=Path,
        help="Path to an LTX-2.3 FLF2V workflow (API format)",
    )
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width", type=int, default=1280, help="Must be divisible by 32")
    parser.add_argument("--height", type=int, default=704, help="Must be divisible by 32")
    parser.add_argument(
        "--first-strength",
        type=float,
        default=1.0,
        help="First-frame anchor strength (default 1.0)",
    )
    parser.add_argument(
        "--last-strength",
        type=float,
        default=0.3,
        help="Last-frame anchor strength (default 0.3). Lower permits more motion.",
    )
    parser.add_argument("--scene", default=None, help="Run only the named scene id")
    parser.add_argument(
        "--comfy-url",
        default=DEFAULT_COMFY_URL,
        help="ComfyUI REST base URL (env: COMFY_URL)",
    )
    parser.add_argument(
        "--comfy-input-dir",
        type=Path,
        default=DEFAULT_COMFY_INPUT_DIR,
        help="Directory to stage keyframes into (env: COMFY_INPUT_DIR)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.width % 32 or args.height % 32:
        raise SystemExit("--width and --height must each be divisible by 32")

    plan = json.loads(args.plan.read_text())
    root = args.plan.parent
    story_id = root.name
    clips_dir = root / "clips"
    clips_dir.mkdir(exist_ok=True)

    workflow_template = json.loads(args.workflow.read_text())
    client = ComfyClient(args.comfy_url)

    scenes = [
        s for s in plan.get("scenes", [])
        if args.scene is None or s["id"] == args.scene
    ]
    if not scenes:
        raise SystemExit("no scenes matched")

    log(f"=== LTX-2.3 FLF2V: {plan.get('title', story_id)} ({len(scenes)} scenes) ===")
    log(
        f"    defaults first_strength={args.first_strength} "
        f"last_strength={args.last_strength} {args.width}x{args.height}@{args.fps}"
    )

    for scene in scenes:
        scene_id = scene["id"]
        output = clips_dir / f"{scene_id}.mp4"
        if output.exists() and output.stat().st_size > MIN_CLIP_BYTES:
            log(f"  {scene_id}: skip (exists)")
            continue

        keyframe = root / "keyframes" / f"{scene_id}.png"
        if not keyframe.exists():
            log(f"  {scene_id}: skip (no keyframe at {keyframe})")
            continue

        prompt = scene.get("prompt", "")
        if not prompt:
            log(f"  {scene_id}: skip (empty prompt)")
            continue

        run_scene(
            client,
            workflow_template,
            keyframe=keyframe,
            output=output,
            comfy_input_dir=args.comfy_input_dir,
            scene_id=scene_id,
            story_id=story_id,
            prompt=prompt,
            width=args.width,
            height=args.height,
            duration=int(scene.get("duration_s", 3)),
            fps=args.fps,
            first_strength=float(scene.get("first_strength", args.first_strength)),
            last_strength=float(scene.get("last_strength", args.last_strength)),
        )

    log(f"=== clips written to {clips_dir} ===")


if __name__ == "__main__":
    main()
