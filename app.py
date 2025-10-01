import asyncio
import argparse
import logging
import os
import tempfile
import time
import uuid
from collections import deque
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from typing import Any, Deque, Dict, List, Optional

import dotenv
import ffmpeg
import ivrit
import magic
import posthog
import uvicorn
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from werkzeug.utils import secure_filename


dotenv.load_dotenv()

parser = argparse.ArgumentParser(description="Local transcription service")
parser.add_argument("--staging", action="store_true", help="Enable staging mode")
parser.add_argument("--hiatus", action="store_true", help="Enable hiatus mode")
parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
parser.add_argument("--dev", action="store_true", help="Enable development mode")
args, _ = parser.parse_known_args()

in_dev = args.staging or args.dev
in_hiatus_mode = args.hiatus
verbose = args.verbose

MODEL_NAME = os.environ.get("MODEL_NAME", "ivrit-ai/whisper-large-v3-turbo-ct2")
MODEL_DEVICE = os.environ.get("MODEL_DEVICE")
ESTIMATED_REALTIME_FACTOR = float(os.environ.get("LOCAL_TRANSCRIBE_RTF", "2.5"))
MAX_AUDIO_DURATION_IN_HOURS = float(os.environ.get("MAX_AUDIO_DURATION_IN_HOURS", "20"))
MAX_FILE_SIZE_BYTES = int(os.environ.get("MAX_FILE_SIZE_BYTES", str(300 * 1024 * 1024)))
RESULT_EXPIRY_MINUTES = int(os.environ.get("RESULT_EXPIRY_MINUTES", "60"))
FILE_CHUNK_SIZE = 16 * 1024 * 1024  # 16MB


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


ENABLE_DIARIZATION = _env_flag("ENABLE_DIARIZATION", "1")
ENABLE_WORD_TIMESTAMPS = _env_flag("ENABLE_WORD_TIMESTAMPS", "1")

app = FastAPI(title="Transcription Service", version="2.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)
if verbose:
    logger.setLevel(logging.DEBUG)

file_handler = RotatingFileHandler(
    filename="app.log", maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
)
file_handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
logger.addHandler(file_handler)

templates = Jinja2Templates(directory="templates")

ph = None
if "POSTHOG_API_KEY" in os.environ:
    ph = posthog.Posthog(project_api_key=os.environ["POSTHOG_API_KEY"], host="https://us.i.posthog.com")

def log_message(message: str) -> None:
    logger.info(message)


def capture_event(distinct_id: str, event: str, props: Optional[Dict[str, Any]] = None) -> None:
    if not ph:
        return
    props = {} if not props else props
    props["source"] = "transcribe.ivrit.ai"
    ph.capture(distinct_id=distinct_id, event=event, properties=props)


ffmpeg_supported_mimes = [
    "video/",
    "audio/",
    "application/mp4",
    "application/x-matroska",
    "application/mxf",
]


def is_ffmpeg_supported_mimetype(file_mime: str) -> bool:
    return any(file_mime.startswith(supported_mime) for supported_mime in ffmpeg_supported_mimes)


def get_media_duration(file_path: str) -> Optional[float]:
    try:
        probe = ffmpeg.probe(file_path)
        audio_info = next(s for s in probe["streams"] if s["codec_type"] == "audio")
        return float(audio_info.get("duration", 0.0))
    except (ffmpeg.Error, StopIteration) as error:
        log_message(f"Error getting media duration: {error}")
        return None


def clean_some_unicode_from_text(text: str) -> str:
    chars_to_remove = "\u061C\u200B\u200C\u200D\u200E\u200F\u202A\u202B\u202C\u202D\u202E\u2066\u2067\u2068\u2069\uFEFF"
    return text.translate({ord(c): None for c in chars_to_remove})


def human_readable_size(num_bytes: int) -> str:
    for unit in ["Bytes", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024 or unit == "TB":
            return f"{num_bytes:.0f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.0f} TB"


temp_files: Dict[str, str] = {}
job_results: Dict[str, Dict[str, Any]] = {}
pending_jobs: Deque[str] = deque()
model_lock = asyncio.Lock()
transcribe_lock = asyncio.Lock()
model_instance: Optional[Any] = None
current_job_id: Optional[str] = None


def cleanup_temp_file(job_id: str) -> None:
    path = temp_files.pop(job_id, None)
    if path and os.path.exists(path):
        try:
            os.unlink(path)
        except OSError as error:
            log_message(f"Error deleting temporary file {path}: {error}")


def cleanup_finished_jobs() -> None:
    if not job_results:
        return
    expiry_time = datetime.now() - timedelta(minutes=RESULT_EXPIRY_MINUTES)
    to_delete = [jid for jid, data in job_results.items() if data.get("completion_time") and data["completion_time"] < expiry_time]
    for job_id in to_delete:
        cleanup_temp_file(job_id)
        job_results.pop(job_id, None)


def normalize_segments(raw_segments: Optional[List[Any]]) -> List[Dict[str, Any]]:
    if not raw_segments:
        return []

    def get_attr(item: Any, key: str, default: Any = None) -> Any:
        if isinstance(item, dict):
            return item.get(key, default)
        return getattr(item, key, default)

    normalized: List[Dict[str, Any]] = []
    for segment in raw_segments:
        start = float(get_attr(segment, "start", 0.0) or 0.0)
        end = float(get_attr(segment, "end", start) or start)
        speakers = get_attr(segment, "speakers", None) or []
        speaker = get_attr(segment, "speaker", None)
        if not speakers and speaker is not None:
            speakers = [f"SPEAKER_{int(speaker)}" if isinstance(speaker, (int, float)) else str(speaker)]

        words_data = get_attr(segment, "words", None) or []
        normalized_words: List[Dict[str, Any]] = []
        for word in words_data:
            word_start = float(get_attr(word, "start", start) or start)
            word_end = float(get_attr(word, "end", word_start) or word_start)
            word_text = get_attr(word, "word", None) or get_attr(word, "text", "")
            normalized_words.append(
                {
                    "start": word_start,
                    "end": word_end,
                    "word": clean_some_unicode_from_text(str(word_text)),
                    "confidence": get_attr(word, "confidence", None),
                }
            )

        normalized.append(
            {
                "id": get_attr(segment, "id", None),
                "start": start,
                "end": end,
                "text": clean_some_unicode_from_text(str(get_attr(segment, "text", ""))),
                "speakers": speakers,
                "words": normalized_words,
            }
        )

    return normalized


async def get_transcription_model():
    global model_instance
    if model_instance is not None:
        return model_instance

    async with model_lock:
        if model_instance is None:
            def load_model() -> Any:
                kwargs: Dict[str, Any] = {"engine": "faster-whisper", "model": MODEL_NAME}
                if MODEL_DEVICE:
                    kwargs["device"] = MODEL_DEVICE
                return ivrit.load_model(**kwargs)

            model_instance = await run_in_threadpool(load_model)
            log_message(f"Loaded ivrit model {MODEL_NAME} (device={MODEL_DEVICE or 'auto'})")
    return model_instance


def estimate_queue_eta(job_id: str) -> str:
    seconds = 0.0
    if current_job_id and current_job_id != job_id:
        active_job = job_results.get(current_job_id)
        if active_job and active_job.get("status") == "running":
            duration = float(active_job.get("duration") or 0.0)
            start_time = active_job.get("start_time")
            if start_time:
                elapsed = time.time() - start_time
                total = max(duration / ESTIMATED_REALTIME_FACTOR, 1.0)
                seconds += max(total - elapsed, 0.0)

    if job_id in pending_jobs:
        ahead = list(pending_jobs)
        index = ahead.index(job_id)
        for queued_job_id in ahead[:index]:
            queued_job = job_results.get(queued_job_id)
            if queued_job:
                duration = float(queued_job.get("duration") or 0.0)
                seconds += max(duration / ESTIMATED_REALTIME_FACTOR, 1.0)

    return str(timedelta(seconds=int(max(seconds, 0))))


def update_running_progress(job_data: Dict[str, Any]) -> float:
    start_time = job_data.get("start_time")
    duration = float(job_data.get("duration") or 0.0)
    if not start_time or duration <= 0:
        return job_data.get("progress", 0.0)

    elapsed = time.time() - start_time
    estimated_total = max(duration / ESTIMATED_REALTIME_FACTOR, 1.0)
    progress = min(elapsed / estimated_total, 0.95)
    job_data["progress"] = progress
    return progress


async def transcribe_job(job_id: str) -> None:
    global current_job_id
    job_data = job_results.get(job_id)
    if not job_data:
        return

    async with transcribe_lock:
        job_data["status"] = "running"
        job_data["start_time"] = time.time()
        try:
            pending_jobs.remove(job_id)
        except ValueError:
            pass

        current_job_id = job_id
        capture_event(
            job_id,
            "transcribe-start",
            {
                "filename": job_data.get("filename"),
                "queue_wait_seconds": job_data["start_time"] - job_data["created_at"].timestamp(),
                "duration_seconds": job_data.get("duration"),
            },
        )

        try:
            model = await get_transcription_model()

            def run_transcription() -> Any:
                kwargs: Dict[str, Any] = {
                    "path": temp_files[job_id],
                    "language": "he",
                    "verbose": True
                }
                if ENABLE_DIARIZATION:
                    kwargs["diarize"] = True
                    # kwargs["diarization_args"] = {
                    #     "engine": "pyannote",
                    # }
                if ENABLE_WORD_TIMESTAMPS:
                    kwargs["word_timestamps"] = True
                try:
                    return model.transcribe(**kwargs)
                except TypeError:
                    kwargs.pop("diarize", None)
                    kwargs.pop("word_timestamps", None)
                    return model.transcribe(**kwargs)

            result = await run_in_threadpool(run_transcription)
            segments_source = None
            full_text = ""
            if isinstance(result, dict):
                segments_source = result.get("segments")
                full_text = str(result.get("text", ""))
            else:
                segments_source = getattr(result, "segments", None)
                full_text = str(getattr(result, "text", ""))

            job_data["results"] = normalize_segments(segments_source)
            job_data["text"] = clean_some_unicode_from_text(full_text)
            job_data["progress"] = 1.0
            job_data["status"] = "done"
            job_data["completion_time"] = datetime.now()
            capture_event(
                job_id,
                "transcribe-done",
                {
                    "duration_seconds": job_data.get("duration"),
                    "segment_count": len(job_data["results"]),
                },
            )
        except Exception as error:  # pylint: disable=broad-except
            log_message(f"Error in transcription job {job_id}: {error}")
            job_data["error"] = "אירעה שגיאה בתהליך התמלול."
            job_data["status"] = "error"
            job_data["progress"] = 1.0
            capture_event(job_id, "transcribe-failed", {"error": str(error)})
        finally:
            cleanup_temp_file(job_id)
            current_job_id = None
            cleanup_finished_jobs()


@app.get("/")
async def index(request: Request):
    if in_hiatus_mode:
        return templates.TemplateResponse("server-down.html", {"request": request})
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...)):
    cleanup_finished_jobs()

    job_id = str(uuid.uuid4())
    if in_hiatus_mode:
        capture_event(job_id, "file-upload-hiatus-rejected")
        return JSONResponse({"error": "השירות כרגע לא פעיל. אנא נסה שוב מאוחר יותר."}, status_code=503)

    if not file or not file.filename:
        return JSONResponse({"error": "לא נבחר קובץ. אנא בחר קובץ להעלאה."}, status_code=400)

    filename = secure_filename(file.filename)
    capture_event(job_id, "file-upload", {"filename": filename})

    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_FILE_SIZE_BYTES:
        return JSONResponse(
            {
                "error": f"הקובץ גדול מדי. הגודל המקסימלי המותר הוא {human_readable_size(MAX_FILE_SIZE_BYTES)}."
            },
            status_code=400,
        )

    temp = tempfile.NamedTemporaryFile(delete=False)
    temp_path = temp.name
    temp.close()

    total_size = 0
    try:
        with open(temp_path, "wb") as buffer:
            while chunk := await file.read(FILE_CHUNK_SIZE):
                total_size += len(chunk)
                if total_size > MAX_FILE_SIZE_BYTES:
                    os.unlink(temp_path)
                    return JSONResponse(
                        {
                            "error": f"הקובץ גדול מדי. הגודל המקסימלי המותר הוא {human_readable_size(MAX_FILE_SIZE_BYTES)}."
                        },
                        status_code=400,
                    )
                buffer.write(chunk)
    except Exception as error:  # pylint: disable=broad-except
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        log_message(f"File upload failed: {error}")
        return JSONResponse({"error": "העלאת הקובץ נכשלה. אנא נסה שוב."}, status_code=500)
    finally:
        await file.close()

    filetype = magic.Magic(mime=True).from_file(temp_path)
    if not is_ffmpeg_supported_mimetype(filetype):
        os.unlink(temp_path)
        return JSONResponse(
            {"error": f"סוג הקובץ {filetype} אינו נתמך. אנא העלה קובץ אודיו או וידאו תקין."},
            status_code=400,
        )

    duration = get_media_duration(temp_path)
    if duration is None:
        os.unlink(temp_path)
        return JSONResponse(
            {"error": "לא ניתן לקרוא את משך הקובץ. אנא ודא שהקובץ תקין ונסה שוב."},
            status_code=400,
        )

    if duration > MAX_AUDIO_DURATION_IN_HOURS * 3600:
        os.unlink(temp_path)
        return JSONResponse(
            {
                "error": f"הקובץ ארוך מדי. המשך המקסימלי המותר הוא {MAX_AUDIO_DURATION_IN_HOURS:.1f} שעות."
            },
            status_code=400,
        )

    temp_files[job_id] = temp_path

    job_results[job_id] = {
        "id": job_id,
        "filename": filename,
        "created_at": datetime.now(),
        "start_time": None,
        "completion_time": None,
        "duration": duration,
        "progress": 0.0,
        "status": "queued",
        "results": [],
        "text": "",
        "error": None,
    }

    pending_jobs.append(job_id)
    capture_event(job_id, "job-queued", {"queue_size": len(pending_jobs)})

    asyncio.create_task(transcribe_job(job_id))

    return JSONResponse({"job_id": job_id})


@app.get("/job_status/{job_id}")
async def job_status(job_id: str):
    job_data = job_results.get(job_id)
    if not job_data:
        return JSONResponse({"error": "מזהה העבודה אינו תקין. אנא נסה להעלות את הקובץ מחדש."}, status_code=400)

    if job_data.get("error"):
        error_message = job_data["error"]
        job_results.pop(job_id, None)
        return JSONResponse({"error": error_message}, status_code=500)

    status = job_data.get("status")
    if status == "queued":
        position = None
        if job_id in pending_jobs:
            position = list(pending_jobs).index(job_id) + 1
        return JSONResponse(
            {
                "queue_position": position or 1,
                "time_ahead": estimate_queue_eta(job_id),
            }
        )

    if status == "running":
        progress = update_running_progress(job_data)
        return JSONResponse({"progress": progress})

    if status == "done":
        response = {
            "progress": 1.0,
            "results": job_data.get("results", []),
            "completion_time": job_data.get("completion_time").isoformat()
            if job_data.get("completion_time")
            else None,
            "text": job_data.get("text", ""),
        }
        return JSONResponse(response)

    return JSONResponse({"error": "העבודה נכשלה."}, status_code=500)


if __name__ == "__main__":
    port = 4600 if in_dev else 4500
    uvicorn.run(app, host="0.0.0.0", port=port)
