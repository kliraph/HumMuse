"""End-to-end experiment logger writing into artifact and run stores."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from backend.artifact_store import ArtifactStore
from backend.run_store import RunStore
from shared.schemas import ArtifactRef


class ExperimentLogger:
    def __init__(
        self,
        artifact_store: ArtifactStore | None = None,
        run_store: RunStore | None = None,
    ) -> None:
        self.artifact_store = artifact_store or ArtifactStore()
        self.run_store = run_store or RunStore()
        self._started_at: dict[str, float] = {}

    def start_run(self, module_name: str, metadata: dict[str, Any] | None = None) -> str:
        run_id = self.run_store.start_run(module_name, metadata=metadata)
        self._started_at[run_id] = time.perf_counter()
        return run_id

    def log_artifact(
        self,
        run_id: str,
        *,
        kind: str,
        filename: str,
        payload: bytes | str | dict[str, Any],
        base_url: str | None = None,
    ) -> ArtifactRef:
        artifact_id = str(uuid.uuid4())

        if isinstance(payload, bytes):
            path = self.artifact_store.save_bytes(artifact_id, filename, payload)
        elif isinstance(payload, dict):
            path = self.artifact_store.save_json(artifact_id, filename, payload)
        else:
            path = self.artifact_store.save_text(artifact_id, filename, payload)

        relative_path = str(path.relative_to(self.artifact_store.root_dir.parent))
        self.run_store.link_artifact(run_id, artifact_id, kind, relative_path)

        return ArtifactRef(
            artifact_id=artifact_id,
            kind=kind,
            path=str(path),
            url=self.artifact_store.get_artifact_url(artifact_id, filename, base_url=base_url),
        )

    def finish_run(self, run_id: str, error: Exception | None = None) -> None:
        started = self._started_at.pop(run_id, None)
        duration_ms = None
        if started is not None:
            duration_ms = (time.perf_counter() - started) * 1000
        self.run_store.finish_run(
            run_id,
            status="failed" if error else "finished",
            duration_ms=duration_ms,
            error=str(error) if error else None,
        )

    def log_payload_text(self, run_id: str, filename: str, payload: Any) -> ArtifactRef:
        as_text = json.dumps(payload, ensure_ascii=False, indent=2)
        return self.log_artifact(run_id, kind="json_text", filename=filename, payload=as_text)
