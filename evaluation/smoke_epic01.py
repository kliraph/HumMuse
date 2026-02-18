"""Smoke checks for EPIC 0-1 acceptance criteria."""

from __future__ import annotations

import os
import shutil

from backend.artifact_store import ArtifactStore
from backend.experiment_logger import ExperimentLogger
from backend.run_store import RunStore
from shared.schemas import GenerationRequest


def reset_storage() -> None:
    if os.path.exists("storage"):
        shutil.rmtree("storage")
    os.makedirs("storage", exist_ok=True)


def run() -> None:
    reset_storage()

    # Shared schemas validation check.
    req = GenerationRequest(
        prompt="Indie pop chorus",
        melody=[{"pitch": "C4", "start_beat": 0, "duration_beats": 1}],
        preferred_chords=["Am", "F", "C", "G"],
    )
    assert req.prompt == "Indie pop chorus"

    artifact_store = ArtifactStore()
    run_store = RunStore()
    logger = ExperimentLogger(artifact_store=artifact_store, run_store=run_store)

    run_id = logger.start_run("smoke", metadata={"scenario": "epic01"})

    midi_bytes = b"MThd\x00\x00\x00\x06\x00\x00\x00\x01\x00\x60"
    midi_ref = logger.log_artifact(
        run_id,
        kind="midi",
        filename="melody.mid",
        payload=midi_bytes,
    )
    assert artifact_store.read_bytes(midi_ref.artifact_id, "melody.mid") == midi_bytes

    payload = {"tempo": 120, "scale": "C major", "chords": ["C", "G", "Am", "F"]}
    json_ref = logger.log_artifact(
        run_id,
        kind="metadata",
        filename="analysis.json",
        payload=payload,
    )
    assert artifact_store.read_json(json_ref.artifact_id, "analysis.json") == payload

    logger.finish_run(run_id)

    run_row = run_store.get_run(run_id)
    artifacts = run_store.list_artifacts(run_id)
    assert run_row is not None
    assert run_row["status"] == "finished"
    assert len(artifacts) == 2
    print("EPIC 0-1 smoke checks passed")


if __name__ == "__main__":
    run()

