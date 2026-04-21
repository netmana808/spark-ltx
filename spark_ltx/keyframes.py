"""Batch driver for FLUX.2 keyframe generation against a local ComfyUI server.

Reads a ``plan.json`` and generates one keyframe PNG per scene into
``<plan_dir>/keyframes/<id>.png``. Uses a FLUX.2 text-to-image workflow
(typically the GGUF Unsloth variant) and patches it per-scene by matching
ComfyUI node ``class_type`` + ``_meta.title`` so the patcher survives
template edits and re-saves.

Plan-level options drive prompt construction:
  style.tokens          — string prefix prepended to every scene prompt
  characters.<name>     — character spec with optional identity_lock string
  scenes[].keyframe_prompt — the Flux image prompt for this scene
  scenes[].characters_in_scene — character names whose identity_lock to inject
  scenes[].negative_prompt — per-scene override (currently unused; T2I workflow
                             has no negative prompt node)
"""
from __future__ import annotations

import argparse
import json
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from .comfy import ComfyClient, IMAGE_EXTS

DEFAULT_COMFY_URL = os.environ.get("COMFY_URL", "http://localhost:8188")
MIN_IMAGE_BYTES = 5_000


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_prompt(scene: dict[str, Any], plan: dict[str, Any]) -> str:
    """Compose the final Flux prompt: style tokens + character identity locks + scene prompt."""
    parts: list[str] = []

    tokens = plan.get("style", {}).get("tokens")
    if tokens:
        parts.append(tokens if isinstance(tokens, str) else ", ".join(tokens))

    characters = plan.get("characters", {}) or {}
    for name in scene.get("characters_in_scene", []):
        lock = characters.get(name, {}).get("identity_lock")
        if lock:
            parts.append(lock)

    scene_prompt = scene.get("keyframe_prompt") or scene.get("prompt") or ""
    if scene_prompt:
        parts.append(scene_prompt)

    return ", ".join(p for p in parts if p)


def derive_seed(scene_id: str, base: int = 501, step: int = 137) -> int:
    """Deterministic seed per scene so re-runs are reproducible."""
    numeric = "".join(c for c in scene_id if c.isdigit()) or "0"
    return int(numeric) * step + base


def patch_workflow(
    workflow: dict[str, Any],
    *,
    prompt: str,
    width: int,
    height: int,
    steps: int,
    seed: int,
    filename_prefix: str,
) -> None:
    """Patch a FLUX.2 T2I workflow in place.

    Matches nodes by ``class_type`` + ``_meta.title`` so numeric node IDs
    can change without breaking the patcher.
    """
    for node in workflow.values():
        class_type = node.get("class_type", "")
        title = node.get("_meta", {}).get("title", "")
        inputs = node.setdefault("inputs", {})

        if class_type == "CLIPTextEncode" and "Positive" in title:
            inputs["text"] = prompt
        elif class_type == "EmptyFlux2LatentImage":
            inputs["width"] = width
            inputs["height"] = height
        elif class_type == "Flux2Scheduler":
            inputs["width"] = width
            inputs["height"] = height
            inputs["steps"] = steps
        elif class_type == "RandomNoise":
            inputs["noise_seed"] = seed
        elif class_type == "SaveImage":
            inputs["filename_prefix"] = filename_prefix


def run_scene(
    client: ComfyClient,
    workflow_template: dict[str, Any],
    *,
    plan: dict[str, Any],
    scene: dict[str, Any],
    output: Path,
    width: int,
    height: int,
    steps: int,
    story_id: str,
) -> bool:
    scene_id = scene["id"]
    prompt = build_prompt(scene, plan)
    if not prompt:
        log(f"  {scene_id}: skip (empty prompt)")
        return False

    seed = derive_seed(scene_id)
    filename_prefix = f"spark_ltx/{story_id}/kf_{scene_id}"
    workflow = deepcopy(workflow_template)
    patch_workflow(
        workflow,
        prompt=prompt,
        width=width,
        height=height,
        steps=steps,
        seed=seed,
        filename_prefix=filename_prefix,
    )

    log(f"  {scene_id}: submit seed={seed} prompt={prompt[:80]!r}")
    started = time.time()
    prompt_id = client.submit(workflow)
    try:
        result = client.poll(prompt_id)
    except (TimeoutError, RuntimeError) as exc:
        log(f"  {scene_id}: FAILED — {exc}")
        return False

    ok, size = client.download(result.get("outputs", {}), output, extensions=IMAGE_EXTS)
    elapsed = time.time() - started
    if ok and size >= MIN_IMAGE_BYTES:
        log(f"  {scene_id}: OK {elapsed:.0f}s {size / 1024:.0f} KB")
        return True
    log(f"  {scene_id}: no image after {elapsed:.0f}s (size={size})")
    return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch-generate FLUX.2 keyframes for every scene in a plan.json.",
    )
    parser.add_argument("plan", type=Path, help="Path to plan.json")
    parser.add_argument(
        "workflow",
        type=Path,
        help="Path to a FLUX.2 T2I workflow (API format)",
    )
    parser.add_argument("--width", type=int, default=1024, help="Must be divisible by 16")
    parser.add_argument("--height", type=int, default=1024, help="Must be divisible by 16")
    parser.add_argument(
        "--steps",
        type=int,
        default=28,
        help="Flux2Scheduler step count (default 28)",
    )
    parser.add_argument("--scene", default=None, help="Generate only this scene id")
    parser.add_argument(
        "--comfy-url",
        default=DEFAULT_COMFY_URL,
        help="ComfyUI REST base URL (env: COMFY_URL)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.width % 16 or args.height % 16:
        raise SystemExit("--width and --height must each be divisible by 16")

    plan = json.loads(args.plan.read_text())
    root = args.plan.parent
    story_id = root.name
    keyframes_dir = root / "keyframes"
    keyframes_dir.mkdir(exist_ok=True)

    workflow_template = json.loads(args.workflow.read_text())
    client = ComfyClient(args.comfy_url)

    scenes = [
        s for s in plan.get("scenes", [])
        if args.scene is None or s["id"] == args.scene
    ]
    if not scenes:
        raise SystemExit("no scenes matched")

    log(f"=== FLUX.2 keyframes: {plan.get('title', story_id)} ({len(scenes)} scenes) ===")
    log(f"    {args.width}x{args.height} steps={args.steps}")

    done = 0
    for scene in scenes:
        scene_id = scene["id"]
        output = keyframes_dir / f"{scene_id}.png"
        if output.exists() and output.stat().st_size >= MIN_IMAGE_BYTES:
            log(f"  {scene_id}: skip (exists)")
            done += 1
            continue

        ok = run_scene(
            client,
            workflow_template,
            plan=plan,
            scene=scene,
            output=output,
            width=args.width,
            height=args.height,
            steps=args.steps,
            story_id=story_id,
        )
        if ok:
            done += 1

    log(f"=== {done}/{len(scenes)} keyframes written to {keyframes_dir} ===")


if __name__ == "__main__":
    main()
