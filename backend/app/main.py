from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from threading import Lock
from typing import Optional

import torch

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel

from app.srt_utils import segments_to_srt

BASE_DIR = Path(__file__).resolve().parent.parent
TMP_DIR = BASE_DIR / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Subtitle Tool API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model_cache: dict[str, WhisperModel] = {}
_model_lock = Lock()
_qwen_model = None
_qwen_model_lock = Lock()


def _normalize_qwen_language(language: Optional[str]) -> Optional[str]:
    if not language:
        return None

    normalized = language.strip().lower()
    mapping = {
        "en": "English",
        "english": "English",
        "zh": "Chinese",
        "zh-cn": "Chinese",
        "zh-hans": "Chinese",
        "zh-hant": "Chinese",
        "chinese": "Chinese",
    }
    return mapping.get(normalized)


def _get_model(model_size: str) -> WhisperModel:
    with _model_lock:
        cached = _model_cache.get(model_size)
        if cached is not None:
            return cached

        # Int8 compute keeps local CPU usage practical for development.
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        _model_cache[model_size] = model
        return model


def _extract_audio_to_wav(video_path: Path, audio_path: Path) -> None:
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(audio_path),
    ]

    try:
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail="FFmpeg not found on system PATH.") from exc
    except subprocess.CalledProcessError as exc:
        raise HTTPException(status_code=400, detail=f"FFmpeg failed: {exc.stderr.strip()}") from exc


def _get_qwen_model():
    global _qwen_model

    with _qwen_model_lock:
        if _qwen_model is not None:
            return _qwen_model

        try:
            from qwen_asr import Qwen3ASRModel
        except ImportError as exc:
            raise HTTPException(
                status_code=500,
                detail=(
                    "qwen-asr is not installed. Install dependencies and restart backend. "
                    "Expected package: qwen-asr"
                ),
            ) from exc

        use_cuda = torch.cuda.is_available()
        use_mps = torch.backends.mps.is_available()
        dtype = torch.bfloat16 if use_cuda else torch.float32
        device_map = "mps" if use_mps else "cpu"

        _qwen_model = Qwen3ASRModel.from_pretrained(
            "Qwen/Qwen3-ASR-0.6B",
            dtype=dtype,
            device_map=device_map,
            max_inference_batch_size=1,
            max_new_tokens=1024,
            forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
            forced_aligner_kwargs=dict(
                dtype=torch.bfloat16,
                device_map=device_map,
                # attn_implementation="flash_attention_2",
            ),
        )
        return _qwen_model


def _transcribe_with_qwen(audio_path: Path, language: Optional[str]) -> tuple[list[dict], str, float]:
    qwen_model = _get_qwen_model()
    qwen_language = _normalize_qwen_language(language)

    try:
        results = qwen_model.transcribe(
            audio=str(audio_path),
            language=qwen_language,
            return_time_stamps=True,
        )
    except TypeError:
        # Fallback for older qwen-asr versions that don't accept timestamp toggle.
        results = qwen_model.transcribe(
            audio=str(audio_path),
            language=qwen_language,
        )

    if not results:
        raise HTTPException(status_code=500, detail="Qwen returned no transcription results.")

    first = results[0]
    text = str(getattr(first, "text", "")).strip()
    detected_language = str(getattr(first, "language", "unknown"))

    if not text:
        raise HTTPException(status_code=500, detail="Qwen returned empty text for this audio.")

    segments = [
        {
            "start": 0.0,
            "end": 0.0,
            "text": text,
        }
    ]
    return segments, detected_language, 1.0


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/transcribe")
async def transcribe_video(
    file: UploadFile = File(...),
    model_size: str = Form("large-v3"),
    language: Optional[str] = Form(None),
) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")

    allowed = {"tiny", "base", "small", "medium", "large-v3", "qwen3-asr-0.6b"}
    if model_size not in allowed:
        raise HTTPException(status_code=400, detail=f"model_size must be one of {sorted(allowed)}")

    safe_name = Path(file.filename).name
    video_path = TMP_DIR / f"input_{os.getpid()}_{safe_name}"
    audio_path = TMP_DIR / f"audio_{os.getpid()}_{video_path.stem}.wav"

    try:
        with video_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        _extract_audio_to_wav(video_path, audio_path)

        if model_size == "qwen3-asr-0.6b":
            segments, detected_language, language_probability = _transcribe_with_qwen(audio_path, language)
        else:
            model = _get_model(model_size)
            whisper_segments, info = model.transcribe(
                str(audio_path),
                language=language,
                vad_filter=True,
                beam_size=5,
            )

            segments = []
            for seg in whisper_segments:
                segments.append(
                    {
                        "start": round(seg.start, 3),
                        "end": round(seg.end, 3),
                        "text": seg.text.strip(),
                    }
                )
            detected_language = info.language
            language_probability = info.language_probability

        srt_content = segments_to_srt(segments)

        return {
            "filename": safe_name,
            "language": detected_language,
            "language_probability": language_probability,
            "model_size": model_size,
            "segments": segments,
            "srt": srt_content,
        }
    finally:
        for path in (video_path, audio_path):
            if path.exists():
                path.unlink()
