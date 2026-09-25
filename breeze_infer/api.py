"""Thin streaming API over the PyTorch Breeze inference runtime."""

from __future__ import annotations

import argparse
import gc
import tempfile
import threading
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from breeze_infer.adapter import MANIFEST_NAME, load_adapter_manifest
from breeze_infer.runtime import (
    load_runtime,
    resolve_device,
    set_all_seeds,
    update_generation_config_for_breeze,
)
from breeze_infer.templates import get_template, prepare_inputs
from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig
from models.warmup_profile import load_warmup_profile

REPO_ROOT = Path(__file__).resolve().parents[1]
FAST_CONFIG = REPO_ROOT / "configs" / "fast.json"
WEB_ROOT = Path(__file__).resolve().parent / "web"
DEFAULT_CFG_SCALE = 1.0
MAX_NEW_TOKENS = 1500
MAX_SEQ_LEN = 2048
REPETITION_PENALTY = 1.1
OPTIONAL_AUDIO_FILE = File(None)


@dataclass(frozen=True)
class ApiSettings:
    model: Path
    adapter: Path | None
    adapters_dir: Path | None
    base_revision: str | None
    fast_all: bool | None
    fast_text_encoder: bool
    fast_backbone_prefill: bool
    fast_backbone_decode: bool
    fast_depth_decoder: bool
    fast_codec: bool


_settings: ApiSettings | None = None
_request_lock = threading.Lock()
_state_lock = threading.Lock()


@dataclass
class ReloadJob:
    id: str
    adapter_id: str
    status: str
    stage: str
    progress: float
    eta_seconds: float | None
    started_at: float
    finished_at: float | None = None
    error: str | None = None


class ReloadRequest(BaseModel):
    adapter_id: str


def _pcm16(audio: np.ndarray) -> bytes:
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767.0).astype("<i2", copy=False).tobytes()


def _resolve_fast_all_default(
    master: bool | None, stage_flags: tuple[bool, ...]
) -> bool | None:
    """Default the API to fast-all without masking explicit stage profiling."""
    if master is not None:
        return master
    return None if any(stage_flags) else True


async def _save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "reference.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(
        prefix="breeze_ref_", suffix=suffix, delete=False
    ) as temporary:
        path = Path(temporary.name)
        try:
            payload = await upload.read()
            if not payload:
                raise HTTPException(status_code=400, detail="Reference audio is empty.")
            temporary.write(payload)
        except Exception:
            path.unlink(missing_ok=True)
            raise
    return path


def _build_runtime(
    settings: ApiSettings,
    adapter: Path | None,
    *,
    progress: Any | None = None,
) -> tuple[Any, Any, Any, FastBreezeStreamingRuntime]:
    tokenizer, model, audio_tokenizer = load_runtime(
        settings.model,
        device=resolve_device(),
        attn_implementation="eager",
        adapter_root=adapter,
        base_revision=settings.base_revision,
        progress=progress,
    )
    update_generation_config_for_breeze(model)

    config = FastStreamingConfig(
        max_new_tokens=MAX_NEW_TOKENS,
        max_seq_len=MAX_SEQ_LEN,
        fast_all=settings.fast_all,
        fast_text_encoder=settings.fast_text_encoder,
        fast_backbone_prefill=settings.fast_backbone_prefill,
        fast_backbone_decode=settings.fast_backbone_decode,
        fast_depth_decoder=settings.fast_depth_decoder,
        fast_codec=settings.fast_codec,
        repetition_penalty=REPETITION_PENALTY,
    )
    runtime = FastBreezeStreamingRuntime(
        model, audio_tokenizer, config, tokenizer=tokenizer
    )
    if runtime.fast_enabled:
        if progress is not None:
            progress("Warming CUDA graphs", 0.90)
        profile = load_warmup_profile(FAST_CONFIG)
        profile = replace(profile, codec_chunk_frames=runtime.codec_chunk_frames)
        manifest = runtime.warmup_from_profile(profile)
        print(f"fast warmup: {manifest['total_elapsed_ms']:.2f} ms", flush=True)

    if progress is not None:
        progress("Finalizing runtime", 0.99)
    return tokenizer, model, audio_tokenizer, runtime


def _adapter_id_for_path(settings: ApiSettings, adapter: Path | None) -> str:
    if adapter is None:
        return "base"
    if settings.adapters_dir is not None:
        try:
            return adapter.resolve().relative_to(settings.adapters_dir.resolve()).as_posix()
        except ValueError:
            pass
    return "startup"


def _load_app(app: FastAPI, settings: ApiSettings) -> None:
    tokenizer, model, audio_tokenizer, runtime = _build_runtime(
        settings, settings.adapter
    )

    app.state.tokenizer = tokenizer
    app.state.model = model
    app.state.audio_tokenizer = audio_tokenizer
    app.state.runtime = runtime
    app.state.active_adapter_id = _adapter_id_for_path(settings, settings.adapter)
    app.state.reload_jobs = {}
    app.state.reload_history = []


def _discover_adapters(settings: ApiSettings) -> list[dict[str, Any]]:
    active_id = getattr(app.state, "active_adapter_id", "base")
    options: list[dict[str, Any]] = [
        {
            "id": "base",
            "name": "Base model",
            "description": "Breeze TTS 2 without a LoRA adapter",
            "active": active_id == "base",
        }
    ]
    seen = {"base"}
    if settings.adapters_dir is not None and settings.adapters_dir.is_dir():
        root = settings.adapters_dir.resolve()
        for manifest_path in sorted(root.rglob(MANIFEST_NAME)):
            adapter_path = manifest_path.parent
            adapter_id = adapter_path.relative_to(root).as_posix()
            try:
                manifest = load_adapter_manifest(adapter_path)
                description = (
                    f"{manifest.variant.replace('_', ' ')} · rank {manifest.rank:g} · "
                    f"alpha {manifest.alpha:g}"
                )
            except Exception as exc:
                description = f"Invalid adapter: {exc}"
            options.append(
                {
                    "id": adapter_id,
                    "name": adapter_path.name,
                    "description": description,
                    "active": adapter_id == active_id,
                }
            )
            seen.add(adapter_id)
    if settings.adapter is not None and "startup" not in seen:
        startup_id = _adapter_id_for_path(settings, settings.adapter)
        if startup_id not in seen:
            options.append(
                {
                    "id": startup_id,
                    "name": settings.adapter.name,
                    "description": "Adapter selected when the server started",
                    "active": startup_id == active_id,
                }
            )
    return options


def _resolve_adapter(settings: ApiSettings, adapter_id: str) -> Path | None:
    if adapter_id == "base":
        return None
    if adapter_id == "startup" and settings.adapter is not None:
        return settings.adapter
    if settings.adapters_dir is None:
        raise HTTPException(status_code=400, detail="No --adapters-dir is configured.")
    root = settings.adapters_dir.resolve()
    candidate = (root / adapter_id).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid adapter id.") from exc
    if not (candidate / MANIFEST_NAME).is_file():
        raise HTTPException(status_code=404, detail="Adapter was not found.")
    return candidate


def _job_payload(job: ReloadJob) -> dict[str, Any]:
    payload = asdict(job)
    payload["progress"] = round(job.progress, 4)
    if job.eta_seconds is not None:
        payload["eta_seconds"] = round(job.eta_seconds, 1)
    payload["elapsed_seconds"] = round(
        (job.finished_at or time.monotonic()) - job.started_at, 1
    )
    return payload


def _update_reload_job(job_id: str, stage: str, progress: float) -> None:
    with _state_lock:
        job = app.state.reload_jobs[job_id]
        job.stage = stage
        job.progress = max(job.progress, min(progress, 0.99))
        elapsed = time.monotonic() - job.started_at
        history = app.state.reload_history
        baseline = sum(history) / len(history) if history else (120.0 if _settings and _settings.fast_all else 60.0)
        estimated_total = max(baseline, elapsed / max(job.progress, 0.02))
        job.eta_seconds = max(0.0, estimated_total - elapsed)


def _dispose_runtime() -> None:
    for name in ("runtime", "audio_tokenizer", "model", "tokenizer"):
        if hasattr(app.state, name):
            setattr(app.state, name, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _reload_worker(job_id: str, adapter_id: str, adapter: Path | None) -> None:
    try:
        _update_reload_job(job_id, "Releasing current weights", 0.04)
        _dispose_runtime()
        assert _settings is not None
        bundle = _build_runtime(
            _settings,
            adapter,
            progress=lambda stage, fraction: _update_reload_job(
                job_id, stage, fraction
            ),
        )
        tokenizer, model, audio_tokenizer, runtime = bundle
        app.state.tokenizer = tokenizer
        app.state.model = model
        app.state.audio_tokenizer = audio_tokenizer
        app.state.runtime = runtime
        app.state.active_adapter_id = adapter_id
        with _state_lock:
            job = app.state.reload_jobs[job_id]
            job.status = "ready"
            job.stage = "Ready"
            job.progress = 1.0
            job.eta_seconds = 0.0
            job.finished_at = time.monotonic()
            app.state.reload_history.append(job.finished_at - job.started_at)
            app.state.reload_history[:] = app.state.reload_history[-5:]
    except Exception as exc:
        with _state_lock:
            job = app.state.reload_jobs[job_id]
            job.status = "failed"
            job.stage = "Reload failed"
            job.error = str(exc)
            job.eta_seconds = None
            job.finished_at = time.monotonic()
    finally:
        _request_lock.release()


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    if _settings is None:
        raise RuntimeError("API settings are not initialized")
    _load_app(app, _settings)
    yield


app = FastAPI(title="Breeze TTS API", lifespan=_lifespan)


@app.get("/health")
def health() -> JSONResponse:
    runtime = getattr(app.state, "runtime", None)
    if runtime is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    return JSONResponse(
        {
            "status": "ok",
            "sample_rate": runtime.sample_rate,
            "adapter_id": getattr(app.state, "active_adapter_id", "base"),
            "fast_path": runtime.fast_enabled,
            "codec_chunk_frames": runtime.codec_chunk_frames,
        }
    )


@app.get("/", include_in_schema=False)
def web_client() -> FileResponse:
    return FileResponse(WEB_ROOT / "index.html", media_type="text/html")


@app.get("/static/{asset}", include_in_schema=False)
def web_asset(asset: str) -> FileResponse:
    if asset not in {"app.css", "app.js"}:
        raise HTTPException(status_code=404, detail="Asset not found.")
    media_type = "text/css" if asset.endswith(".css") else "text/javascript"
    return FileResponse(WEB_ROOT / asset, media_type=media_type)


@app.get("/v1/adapters")
def adapters() -> JSONResponse:
    if _settings is None:
        raise HTTPException(status_code=503, detail="Server is not initialized.")
    return JSONResponse(
        {
            "active_adapter_id": getattr(app.state, "active_adapter_id", "base"),
            "adapters": _discover_adapters(_settings),
        }
    )


@app.post("/v1/adapters/reload", status_code=202)
def reload_adapter(request: ReloadRequest) -> JSONResponse:
    if _settings is None:
        raise HTTPException(status_code=503, detail="Server is not initialized.")
    adapter = _resolve_adapter(_settings, request.adapter_id)
    if not _request_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409, detail="Inference or another weight reload is running."
        )
    job_id = uuid.uuid4().hex
    job = ReloadJob(
        id=job_id,
        adapter_id=request.adapter_id,
        status="loading",
        stage="Queued",
        progress=0.0,
        eta_seconds=None,
        started_at=time.monotonic(),
    )
    with _state_lock:
        app.state.reload_jobs[job_id] = job
        if len(app.state.reload_jobs) > 20:
            oldest = next(iter(app.state.reload_jobs))
            del app.state.reload_jobs[oldest]
    thread = threading.Thread(
        target=_reload_worker,
        args=(job_id, request.adapter_id, adapter),
        daemon=True,
        name=f"breeze-reload-{job_id[:8]}",
    )
    thread.start()
    return JSONResponse(_job_payload(job), status_code=202)


@app.get("/v1/adapters/reload/{job_id}")
def reload_status(job_id: str) -> JSONResponse:
    with _state_lock:
        job = getattr(app.state, "reload_jobs", {}).get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Reload job was not found.")
        return JSONResponse(_job_payload(job))


@app.post("/v1/audio/speech")
async def speech(
    text: str = Form(...),
    instruction: str = Form("Speak clearly and naturally."),
    cfg_scale: float = Form(DEFAULT_CFG_SCALE),
    ref_audio: UploadFile | None = OPTIONAL_AUDIO_FILE,
    ref_text: str = Form(""),
    seed: int = Form(42),
) -> StreamingResponse:
    if not _request_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409, detail="An inference request is already running."
        )

    reference_path: Path | None = None
    try:
        runtime = getattr(app.state, "runtime", None)
        if runtime is None:
            raise HTTPException(status_code=503, detail="Model weights are loading.")
        if not np.isfinite(cfg_scale) or cfg_scale <= 0:
            raise HTTPException(
                status_code=400, detail="cfg_scale must be greater than 0."
            )
        ref_text = ref_text.strip()
        has_reference = ref_audio is not None and bool(ref_audio.filename)
        if has_reference != bool(ref_text):
            raise HTTPException(
                status_code=400,
                detail="ref_audio and ref_text must be provided together or both omitted.",
            )
        if has_reference:
            assert ref_audio is not None
            reference_path = await _save_upload(ref_audio)

        request_id = f"api-{uuid.uuid4().hex}"
        request = {
            "id": request_id,
            "text": text,
            "instruction": instruction,
            "speaker": "S0",
        }
        template_name = "tts_instruction"
        if reference_path is not None:
            request["ref_audio_path"] = str(reference_path)
            request["ref_text"] = ref_text
            template_name = "ref_edit_tata"

        set_all_seeds(seed)
        inputs = prepare_inputs(
            app.state.tokenizer,
            app.state.audio_tokenizer,
            app.state.model,
            [request],
            get_template(template_name),
            guidance_scale=cfg_scale,
            guidance_scale_ref=None,
            guidance_scale_ins=None,
        )
    except Exception:
        if reference_path is not None:
            reference_path.unlink(missing_ok=True)
        _request_lock.release()
        raise

    def body() -> Iterator[bytes]:
        try:
            for chunk in runtime.iter_audio_chunks(
                inputs, request_id=request_id
            ):
                pcm = _pcm16(chunk.audio)
                if pcm:
                    yield pcm
        finally:
            if reference_path is not None:
                reference_path.unlink(missing_ok=True)
            _request_lock.release()

    return StreamingResponse(
        body(),
        media_type="audio/pcm",
        headers={
            "X-Sample-Rate": str(runtime.sample_rate),
            "X-Sample-Format": "s16le",
            "X-Adapter-Id": getattr(app.state, "active_adapter_id", "base"),
            "X-Fast-Path": "1" if runtime.fast_enabled else "0",
            "Cache-Control": "no-store",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve Breeze TTS 2 streaming inference"
    )
    parser.add_argument("model", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument(
        "--adapters-dir",
        type=Path,
        help="Directory containing selectable LoRA adapter folders",
    )
    parser.add_argument(
        "--base-revision",
        help="Exact base revision required by an adapter manifest",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--fast-all",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use the warmed CUDA-graph fast path (default when no stage flags are set)",
    )
    parser.add_argument(
        "--fast-text-encoder", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--fast-backbone-prefill", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--fast-backbone-decode", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--fast-depth-decoder", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--fast-codec", action=argparse.BooleanOptionalAction, default=False
    )
    args = parser.parse_args()

    global _settings
    fast_all = _resolve_fast_all_default(
        args.fast_all,
        (
            args.fast_text_encoder,
            args.fast_backbone_prefill,
            args.fast_backbone_decode,
            args.fast_depth_decoder,
            args.fast_codec,
        ),
    )
    _settings = ApiSettings(
        model=args.model,
        adapter=args.adapter,
        adapters_dir=args.adapters_dir,
        base_revision=args.base_revision,
        fast_all=fast_all,
        fast_text_encoder=args.fast_text_encoder,
        fast_backbone_prefill=args.fast_backbone_prefill,
        fast_backbone_decode=args.fast_backbone_decode,
        fast_depth_decoder=args.fast_depth_decoder,
        fast_codec=args.fast_codec,
    )

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
