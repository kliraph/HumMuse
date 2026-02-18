"""FastAPI app that orchestrates backend generation flows."""

from __future__ import annotations

import mimetypes

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from backend.api_models import (
    LyricsToChordsRequest,
    LyricsToChordsResponse,
    WriterBlockHelpRequest,
    WriterBlockHelpResponse,
)
from backend.experiment_logger import ExperimentLogger
from ml.lyrics_to_chords.service import generate_from_lyrics
from ml.melody_sketchpad.service import generate_from_hum
from ml.writers_block.service import generate_help
from shared.schemas import GenerationResponse

APP_VERSION = "0.2.0"

app = FastAPI(title="HumMuse API", version=APP_VERSION)
logger = ExperimentLogger()


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/version")
def version() -> dict[str, str]:
    return {"version": APP_VERSION}


@app.post("/melody/from-hum", response_model=GenerationResponse)
async def melody_from_hum(
    request: Request,
    audio: UploadFile = File(...),
    prompt: str = Form("Melody sketch from humming"),
    mood: str | None = Form(default=None),
    tempo_bpm: int | None = Form(default=None),
) -> GenerationResponse:
    run_id = logger.start_run(
        "melody_from_hum",
        metadata={"filename": audio.filename, "prompt": prompt, "mood": mood, "tempo_bpm": tempo_bpm},
    )
    try:
        audio_bytes = await audio.read()
        melody, chords, explanation, midi_bytes = generate_from_hum(audio_bytes, tempo_bpm)
        base_url = str(request.base_url).rstrip("/")

        midi_ref = logger.log_artifact(
            run_id,
            kind="midi",
            filename="melody.mid",
            payload=midi_bytes,
            base_url=base_url,
        )
        chord_ref = logger.log_artifact(
            run_id,
            kind="chords",
            filename="chords.json",
            payload={"chords": [c.model_dump() for c in chords]},
            base_url=base_url,
        )
        logger.finish_run(run_id)
        return GenerationResponse(
            run_id=run_id,
            melody=melody,
            chords=chords,
            explanation=explanation,
            artifacts=[midi_ref, chord_ref],
        )
    except Exception as exc:  # pragma: no cover
        logger.finish_run(run_id, error=exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/chords/from-lyrics", response_model=LyricsToChordsResponse)
def chords_endpoint(payload: LyricsToChordsRequest) -> LyricsToChordsResponse:
    mood, top_progressions, explanation = generate_from_lyrics(payload.text)
    return LyricsToChordsResponse(
        mood=mood,
        top_progressions=top_progressions,
        explanation=explanation,
    )


@app.post("/writerblock/help", response_model=WriterBlockHelpResponse)
def writerblock_endpoint(payload: WriterBlockHelpRequest) -> WriterBlockHelpResponse:
    suggestions, explanation = generate_help(payload.text, payload.mood)
    return WriterBlockHelpResponse(suggestions=suggestions, explanation=explanation)


@app.get("/artifact/{artifact_id}/{filename}")
def get_artifact(artifact_id: str, filename: str) -> FileResponse:
    path = logger.artifact_store.get_artifact_path(artifact_id, filename)
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")
    media_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=filename)
