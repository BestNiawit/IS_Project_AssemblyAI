import assemblyai as aai
from flask import Flask, render_template, request, Response
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import os
import time
import uuid
import json
import re
from datetime import datetime, timezone
from collections import defaultdict, deque
import nltk
from transformers import pipeline

app = Flask(__name__, template_folder='templates')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
LEGACY_UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
app.config['UPLOAD_FOLDER'] = UPLOAD_DIR
summarizer = None
translators = {}
MAX_FILE_MB = 50
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_MB * 1024 * 1024
API_KEY_ENV = "ASSEMBLYAI_API_KEY"
RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_MAX_UPLOADS = 5
upload_requests_by_ip = defaultdict(deque)
JOBS_HISTORY_PATH = os.path.join(app.config['UPLOAD_FOLDER'], "jobs_history.json")
TRANSLATION_TARGETS = {
    "thai": {"label": "Thai", "model": "Helsinki-NLP/opus-mt-en-th"},
    "english": {"label": "English", "model": "Helsinki-NLP/opus-mt-th-en"},
}

aai.settings.api_key = "8676fae4821e46e88952da5563a89101"
api_key = (os.getenv(API_KEY_ENV) or "").strip()
if api_key:
    aai.settings.api_key = api_key

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
legacy_jobs_history = os.path.join(LEGACY_UPLOAD_DIR, "jobs_history.json")
if not os.path.exists(JOBS_HISTORY_PATH) and os.path.exists(legacy_jobs_history):
    try:
        os.replace(legacy_jobs_history, JOBS_HISTORY_PATH)
    except Exception as error:
        print(error)


def utc_now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_jobs_history():
    if not os.path.exists(JOBS_HISTORY_PATH):
        return []
    try:
        with open(JOBS_HISTORY_PATH, "r", encoding="utf-8") as file:
            payload = json.load(file)
        return payload if isinstance(payload, list) else []
    except Exception as error:
        print(error)
        return []


def save_jobs_history(jobs):
    temp_path = f"{JOBS_HISTORY_PATH}.tmp"
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(jobs, file, ensure_ascii=False, indent=2)
    os.replace(temp_path, JOBS_HISTORY_PATH)


def append_job_record(record):
    jobs = load_jobs_history()
    jobs.insert(0, record)
    save_jobs_history(jobs)


def find_job_by_id(job_id):
    for job in load_jobs_history():
        if job.get("id") == job_id:
            return job
    return None


def filter_jobs(jobs, query_text, status_filter):
    query_value = (query_text or "").strip().lower()
    status_value = (status_filter or "all").strip().lower()
    filtered = []
    for job in jobs:
        job_status = (job.get("status") or "").lower()
        if status_value != "all" and job_status != status_value:
            continue
        if query_value:
            searchable = " ".join(
                [
                    str(job.get("filename", "")),
                    str(job.get("summary", ""))[:500],
                    str(job.get("transcript", ""))[:500],
                    str(job.get("translated_summary", ""))[:500],
                    str(job.get("translated_transcript", ""))[:500],
                    str(job.get("error_message", "")),
                ]
            ).lower()
            if query_value not in searchable:
                continue
        filtered.append(job)
    return filtered


def text_download_response(filename, content):
    return Response(
        content or "",
        mimetype="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def trim_for_summary(text, max_chars=6000):
    """
    Rough token saver:
    summarizers care about token count; chars is a cheap proxy.
    """
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rsplit(" ", 1)[0] + "..."


def summary_params(summary_length: str):
    """Return (max_chars, max_length, min_length) for summarization."""
    key = (summary_length or "").strip().lower()
    if key == "short":
        return 3000, 120, 20
    if key == "detailed":
        return 9000, 260, 60
    # default: medium
    return 6000, 200, 30


def split_text_for_translation(text, max_chars=900):
    """
    Split large text into small chunks for translation models.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return []
    sentences = re.split(r"(?<=[.!?])\s+|\n+", cleaned)
    chunks = []
    current = []
    current_len = 0
    for sentence in sentences:
        segment = sentence.strip()
        if not segment:
            continue
        segment_len = len(segment)
        if current and current_len + 1 + segment_len > max_chars:
            chunks.append(" ".join(current))
            current = [segment]
            current_len = segment_len
        else:
            current.append(segment)
            current_len += segment_len + (1 if current_len else 0)
    if current:
        chunks.append(" ".join(current))
    return chunks


def get_translator(target_language):
    """
    Lazy-load translation model for requested target language.
    """
    key = (target_language or "").strip().lower()
    config = TRANSLATION_TARGETS.get(key)
    if not config:
        return None

    if key not in translators:
        model_name = config["model"]
        translators[key] = pipeline("translation", model=model_name, tokenizer=model_name)
    return translators[key]


def translate_text(text, target_language):
    translator = get_translator(target_language)
    if translator is None:
        return ""

    chunks = split_text_for_translation(text, max_chars=900)
    if not chunks:
        return ""

    translated_chunks = []
    for chunk in chunks:
        result = translator(chunk, max_length=512)
        translated_chunks.append(result[0]["translation_text"])
    return "\n".join(translated_chunks)


def require_api_key():
    # API key is set in-code (and can still be overridden by env).
    # Keep this as a no-op to avoid blocking uploads with config warnings.
    return None


def get_client_ip():
    forwarded_for = request.headers.get("X-Forwarded-For", "")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.remote_addr or "unknown"


def check_upload_rate_limit():
    """
    Sliding-window rate limit by client IP.
    Returns:
      (limited: bool, retry_after_seconds: int)
    """
    now = time.time()
    ip = get_client_ip()
    request_times = upload_requests_by_ip[ip]

    while request_times and now - request_times[0] > RATE_LIMIT_WINDOW_SECONDS:
        request_times.popleft()

    if len(request_times) >= RATE_LIMIT_MAX_UPLOADS:
        retry_after = max(1, int(RATE_LIMIT_WINDOW_SECONDS - (now - request_times[0])))
        return True, retry_after

    request_times.append(now)
    return False, 0


def build_unique_upload_filename(original_filename):
    """
    Preserve extension but store uploads with UUID names to avoid collisions.
    """
    ext = os.path.splitext(original_filename)[1].lower()
    return f"{uuid.uuid4().hex}{ext}"


def is_empty_upload(file_storage):
    """
    Detect empty uploads safely without consuming the file stream.
    """
    stream = getattr(file_storage, "stream", None)
    if stream is None:
        return True
    current_pos = stream.tell()
    first_byte = stream.read(1)
    stream.seek(current_pos)
    return first_byte == b""


def format_ms(ms):
    """Format milliseconds to HH:MM:SS for readable transcript sections."""
    total_seconds = max(int((ms or 0) / 1000), 0)
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def build_timestamped_transcript(transcript):
    """
    Build a timestamped transcript from utterances when speaker labels are enabled.
    Returns:
      rows: structured rows for UI
      flat_text: plain-text timestamped transcript for copy/download
    """
    rows = []
    flat_lines = []
    utterances = getattr(transcript, "utterances", None) or []

    for utt in utterances:
        text = (getattr(utt, "text", "") or "").strip()
        if not text:
            continue
        speaker = getattr(utt, "speaker", None) or "Unknown"
        start_ms = getattr(utt, "start", 0) or 0
        end_ms = getattr(utt, "end", 0) or 0
        start_label = format_ms(start_ms)
        end_label = format_ms(end_ms)
        rows.append({
            "speaker": speaker,
            "start_label": start_label,
            "end_label": end_label,
            "text": text,
        })
        flat_lines.append(f"[{start_label} - {end_label}] Speaker {speaker}: {text}")

    return rows, "\n".join(flat_lines)


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(_e):
    return render_template('index.html', error_message=f"File too large. Please upload up to {MAX_FILE_MB}MB."), 413


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['wav', 'mp3', 'flac']


def get_summarizer():
    """Lazy-load summarizer only on first use."""
    global summarizer
    if summarizer is None:
        summarizer = pipeline('summarization')
    return summarizer


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/history', strict_slashes=False)
def history():
    query = request.args.get("q", "").strip()
    status = request.args.get("status", "all").strip().lower()
    all_jobs = load_jobs_history()
    jobs = filter_jobs(all_jobs, query, status)
    return render_template(
        "history.html",
        jobs=jobs,
        query=query,
        status=status,
        total_count=len(all_jobs),
        filtered_count=len(jobs),
    )


@app.route('/history/<job_id>', strict_slashes=False)
def history_job(job_id):
    job = find_job_by_id(job_id)
    if not job:
        return render_template('index.html', error_message="Job not found in history."), 404
    return render_template(
        'uploaded_file.html',
        filename=job.get("filename", "Unknown"),
        transcript=job.get("transcript", ""),
        summary=job.get("summary", ""),
        translated_transcript=job.get("translated_transcript", ""),
        translated_summary=job.get("translated_summary", ""),
        translation_target=job.get("translation_target", ""),
        translation_target_label=job.get("translation_target_label", ""),
        timestamped_text=job.get("timestamped_text", ""),
        timestamped_rows=job.get("timestamped_rows", []),
        srt_content=job.get("srt_content", ""),
        vtt_content=job.get("vtt_content", ""),
    )


@app.route('/history/<job_id>/download/<kind>', strict_slashes=False)
def history_download(job_id, kind):
    job = find_job_by_id(job_id)
    if not job:
        return render_template('index.html', error_message="Job not found in history."), 404

    download_map = {
        "transcript": ("transcript.txt", job.get("transcript", "")),
        "summary": ("summary.txt", job.get("summary", "")),
        "translated_transcript": ("translated_transcript.txt", job.get("translated_transcript", "")),
        "translated_summary": ("translated_summary.txt", job.get("translated_summary", "")),
        "timestamped": ("timestamped_transcript.txt", job.get("timestamped_text", "")),
        "srt": ("subtitles.srt", job.get("srt_content", "")),
        "vtt": ("subtitles.vtt", job.get("vtt_content", "")),
    }
    target = download_map.get(kind)
    if target is None:
        return render_template('index.html', error_message="Unsupported download type."), 400

    filename, content = target
    if not content:
        return render_template('index.html', error_message="No content available for this download."), 400
    return text_download_response(filename, content)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        job_id = uuid.uuid4().hex
        # --- Validate request ---
        if 'file' not in request.files:
            return render_template('index.html', error_message="No file provided. Please upload an audio file.")

        file = request.files['file']
        if not file or not getattr(file, "filename", None):
            return render_template('index.html', error_message="No file selected. Please choose an audio file.")

        # Empty filename can happen if the user hits upload without picking a file
        if file.filename.strip() == "":
            return render_template('index.html', error_message="No file selected. Please choose an audio file.")

        if request.content_length and request.content_length > MAX_FILE_MB * 1024 * 1024:
            return render_template('index.html', error_message=f"File too large. Please upload up to {MAX_FILE_MB}MB.")

        filename = secure_filename(file.filename)
        if not allowed_file(filename):
            return render_template(
                'index.html',
                error_message="Unsupported file format. Please upload WAV, MP3, or FLAC."
            )

        if is_empty_upload(file):
            return render_template('index.html', error_message="Uploaded file is empty. Please choose a valid audio file.")

        missing_key_response = require_api_key()
        if missing_key_response:
            return missing_key_response

        limited, retry_after = check_upload_rate_limit()
        if limited:
            return render_template(
                'index.html',
                error_message=f"Too many uploads. Please wait {retry_after}s and try again."
            ), 429

        stored_filename = build_unique_upload_filename(filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], stored_filename)
        try:
            file.save(file_path)
        except Exception as e:
            print(e)
            return render_template('index.html', error_message="Failed to save the uploaded file. Please try again.")

        # --- Transcribe ---
        transcriber = aai.Transcriber()
        transcribe_config = aai.TranscriptionConfig(speaker_labels=True)
        try:
            transcript = transcriber.transcribe(file_path, config=transcribe_config)
        except Exception as e:
            print(e)
            append_job_record({
                "id": job_id,
                "filename": filename,
                "stored_filename": stored_filename,
                "status": "failed",
                "error_message": "Failed to transcribe the audio.",
                "created_at": utc_now_iso(),
            })
            return render_template('index.html', error_message="Failed to transcribe the audio. Please try another file.")

        if transcript is None:
            append_job_record({
                "id": job_id,
                "filename": filename,
                "stored_filename": stored_filename,
                "status": "failed",
                "error_message": "No transcript returned.",
                "created_at": utc_now_iso(),
            })
            return render_template('index.html', error_message="No transcript returned. Please try another audio file.")

        # --- Summarize (lazy-loaded) ---
        current_summarizer = get_summarizer()
        summary_length = request.form.get("summary_length", "medium")
        translation_target = (request.form.get("translation_target", "none") or "none").strip().lower()
        translation_target_label = TRANSLATION_TARGETS.get(translation_target, {}).get("label", "")
        max_chars, max_length, min_length = summary_params(summary_length)

        raw_text = getattr(transcript, "text", None)
        if not isinstance(raw_text, str):
            return render_template('index.html', error_message="Transcription text is missing. Please try another audio file.")

        text = raw_text.strip()
        if not text:
            return render_template('index.html', error_message="Transcription text is empty. Please try another audio file.")

        try:
            # Generate the summary (keep lengths modest to reduce compute/token usage)
            # Token saver: trim long transcripts before summarizing.
            text_for_summary = trim_for_summary(text, max_chars=max_chars)
            summary = current_summarizer(
                text_for_summary,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
            )
            summary_text = summary[0]['summary_text']
        except Exception as e:
            print(e)
            append_job_record({
                "id": job_id,
                "filename": filename,
                "stored_filename": stored_filename,
                "status": "failed",
                "error_message": "Failed to summarize the transcript.",
                "created_at": utc_now_iso(),
                "transcript": text,
            })
            return render_template('index.html', error_message="Failed to summarize the transcript. Please try again with a different file.")

        timestamped_rows, timestamped_text = build_timestamped_transcript(transcript)
        srt_content = ""
        vtt_content = ""
        try:
            srt_content = transcript.export_subtitles_srt()
        except Exception as e:
            print(e)
        try:
            vtt_content = transcript.export_subtitles_vtt()
        except Exception as e:
            print(e)

        translated_transcript = ""
        translated_summary = ""
        translation_error_message = ""
        if translation_target in TRANSLATION_TARGETS:
            try:
                translated_transcript = translate_text(text, translation_target)
                translated_summary = translate_text(summary_text, translation_target)
            except Exception as e:
                print(e)
                translation_error_message = f"Translation to {translation_target_label} failed. Showing original text."

        append_job_record({
            "id": job_id,
            "filename": filename,
            "stored_filename": stored_filename,
            "status": "done",
            "created_at": utc_now_iso(),
            "summary_length": summary_length,
            "translation_target": translation_target,
            "translation_target_label": translation_target_label,
            "transcript": text,
            "summary": summary_text,
            "translated_transcript": translated_transcript,
            "translated_summary": translated_summary,
            "timestamped_rows": timestamped_rows,
            "timestamped_text": timestamped_text,
            "srt_content": srt_content,
            "vtt_content": vtt_content,
        })

        return render_template(
            'uploaded_file.html',
            filename=filename,
            transcript=text,
            summary=summary_text,
            translated_transcript=translated_transcript,
            translated_summary=translated_summary,
            translation_target=translation_target,
            translation_target_label=translation_target_label,
            translation_error_message=translation_error_message,
            timestamped_rows=timestamped_rows,
            timestamped_text=timestamped_text,
            srt_content=srt_content,
            vtt_content=vtt_content
        )
    return render_template('index.html')


@app.route('/uploaded_file/<filename>')
def uploaded_file(filename):
    missing_key_response = require_api_key()
    if missing_key_response:
        return missing_key_response
    try:
        transcriber = aai.Transcriber()
        transcribe_config = aai.TranscriptionConfig(speaker_labels=True)
        transcript = transcriber.transcribe(
            os.path.join(app.config['UPLOAD_FOLDER'], filename),
            config=transcribe_config
        )
        text = getattr(transcript, "text", None) or ""
        timestamped_rows, timestamped_text = build_timestamped_transcript(transcript)
        srt_content = ""
        vtt_content = ""
        try:
            srt_content = transcript.export_subtitles_srt()
        except Exception as e:
            print(e)
        try:
            vtt_content = transcript.export_subtitles_vtt()
        except Exception as e:
            print(e)

        return render_template(
            'uploaded_file.html',
            filename=filename,
            transcript=text,
            timestamped_rows=timestamped_rows,
            timestamped_text=timestamped_text,
            srt_content=srt_content,
            vtt_content=vtt_content
        )
    except Exception as e:
        print(e)
        error_message = "There was an error processing the file. Please check the file format and try again."
        return render_template('index.html', error_message=error_message)


if __name__ == "__main__":
    app.run(debug=True, port=8800, use_reloader=False)
