"""Local artifact storage for experiment outputs."""

from __future__ import annotations

import json
from pathlib import Path


class ArtifactStore:
    def __init__(self, root_dir: str | Path = "storage/artifacts") -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def artifact_dir(self, artifact_id: str) -> Path:
        path = self.root_dir / artifact_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_bytes(self, artifact_id: str, filename: str, data: bytes) -> Path:
        path = self.artifact_dir(artifact_id) / filename
        path.write_bytes(data)
        return path

    def save_json(self, artifact_id: str, filename: str, payload: dict) -> Path:
        path = self.artifact_dir(artifact_id) / filename
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return path

    def save_text(self, artifact_id: str, filename: str, text: str) -> Path:
        path = self.artifact_dir(artifact_id) / filename
        path.write_text(text, encoding="utf-8")
        return path

    def read_bytes(self, artifact_id: str, filename: str) -> bytes:
        return (self.root_dir / artifact_id / filename).read_bytes()

    def read_json(self, artifact_id: str, filename: str) -> dict:
        with (self.root_dir / artifact_id / filename).open("r", encoding="utf-8") as f:
            return json.load(f)

    def read_text(self, artifact_id: str, filename: str) -> str:
        return (self.root_dir / artifact_id / filename).read_text(encoding="utf-8")

    def get_artifact_path(self, artifact_id: str, filename: str) -> Path:
        return self.root_dir / artifact_id / filename

    def get_artifact_url(
        self, artifact_id: str, filename: str, base_url: str | None = None
    ) -> str:
        relative = f"artifact/{artifact_id}/{filename}"
        if base_url:
            return f"{base_url.rstrip('/')}/{relative}"
        return str((self.root_dir / artifact_id / filename).resolve().as_uri())
