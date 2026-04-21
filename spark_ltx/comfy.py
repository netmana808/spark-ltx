"""Minimal ComfyUI REST client: submit a prompt graph, poll until complete, download the output."""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path
from typing import Any

VIDEO_EXTS = (".mp4", ".webm")
_OUTPUT_KEYS = ("video", "videos", "gifs", "images")


class ComfyClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def submit(self, workflow: dict[str, Any]) -> str:
        data = json.dumps({"prompt": workflow}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/prompt",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.load(resp)["prompt_id"]

    def poll(
        self,
        prompt_id: str,
        timeout: float = 1800.0,
        interval: float = 8.0,
    ) -> dict[str, Any]:
        """Block until the prompt has produced outputs, errored, or timed out."""
        deadline = time.time() + timeout
        last_queue_state: tuple[int, int] | None = None
        while time.time() < deadline:
            time.sleep(interval)
            try:
                entry = self._history_entry(prompt_id)
                if entry is not None:
                    outputs = entry.get("outputs") or {}
                    if _has_media(outputs):
                        return entry
                    status = (entry.get("status") or {}).get("status_str")
                    if status == "error":
                        raise RuntimeError(f"prompt {prompt_id} errored")
                queue_state = self._queue_state()
                if queue_state != last_queue_state:
                    running, pending = queue_state
                    print(f"  queue running={running} pending={pending}", flush=True)
                    last_queue_state = queue_state
            except RuntimeError:
                raise
            except Exception as exc:
                print(f"  poll error: {exc}", flush=True)
        raise TimeoutError(f"prompt {prompt_id} timed out after {timeout:.0f}s")

    def download(self, outputs: dict[str, Any], destination: Path) -> tuple[bool, int]:
        """Walk the outputs payload, download the first video artifact, return (ok, bytes_written)."""
        for node_outputs in outputs.values():
            for key in _OUTPUT_KEYS:
                for item in node_outputs.get(key, []):
                    filename = item.get("filename", "")
                    if not filename.endswith(VIDEO_EXTS):
                        continue
                    subfolder = item.get("subfolder", "")
                    file_type = item.get("type", "output")
                    url = (
                        f"{self.base_url}/view"
                        f"?filename={filename}&subfolder={subfolder}&type={file_type}"
                    )
                    with urllib.request.urlopen(url, timeout=120) as resp:
                        data = resp.read()
                    if len(data) > 1000:
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        destination.write_bytes(data)
                        return True, len(data)
        return False, 0

    def _history_entry(self, prompt_id: str) -> dict[str, Any] | None:
        with urllib.request.urlopen(
            f"{self.base_url}/history/{prompt_id}", timeout=15
        ) as resp:
            return json.load(resp).get(prompt_id)

    def _queue_state(self) -> tuple[int, int]:
        with urllib.request.urlopen(f"{self.base_url}/queue", timeout=10) as resp:
            q = json.load(resp)
        return (
            len(q.get("queue_running", [])),
            len(q.get("queue_pending", [])),
        )


def _has_media(outputs: dict[str, Any]) -> bool:
    return any(
        any(key in node_outputs for key in _OUTPUT_KEYS)
        for node_outputs in outputs.values()
    )
