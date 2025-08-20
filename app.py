import os
import re
import time
import shutil
import tempfile
import subprocess
from datetime import datetime
from typing import List, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
import imageio_ffmpeg

# Optional YouTube helpers (declared in requirements)
try:
    import yt_dlp  # type: ignore
except Exception:
    yt_dlp = None  # handled at runtime

try:
    from youtube_transcript_api import (  # type: ignore
        YouTubeTranscriptApi,
        TranscriptsDisabled,
        NoTranscriptFound,
        VideoUnavailable,
    )
except Exception:
    YouTubeTranscriptApi = None  # handled at runtime


# --------------- Theme + Layout ---------------

def inject_width_css(max_width_px: int):
    css = f"""
    <style>
      /* Center and cap width only; use default Streamlit theme */
      .block-container {{
        max-width: {max_width_px}px !important;
        padding-top: 1.0rem;
        padding-bottom: 4rem;
      }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


st.set_page_config(page_title="Video Summarizer — Gemini 2.5 Pro (Files API)", layout="centered")


# --------------- Utilities ---------------

ACCEPTED_EXTS = (".mp4", ".mov", ".mkv", ".webm", ".avi")
MIME_BY_EXT = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
    ".avi": "video/x-msvideo",
}


def human_hms(seconds: int) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def mime_from_ext(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return MIME_BY_EXT.get(ext, "application/octet-stream")


def ffmpeg_exe() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def run_ffmpeg(cmd: List[str]) -> Tuple[int, str, str]:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc.returncode, proc.stdout, proc.stderr


# --------------- Preprocessing ---------------

def preprocess_video(
    input_path: str,
    out_dir: str,
    downscale_720: bool = False,
    cfr_30: bool = False,
    audio_normalize: bool = False,
    mono_downmix: bool = False,
    deinterlace: bool = False,
    trim_silence: bool = False,
) -> str:
    """
    Apply optional preprocessing in a single pass when possible.
    Returns output path if any preprocessing applied; otherwise returns input_path.
    """
    if not any([downscale_720, cfr_30, audio_normalize, mono_downmix, deinterlace, trim_silence]):
        return input_path

    ensure_dir(out_dir)
    base = os.path.splitext(os.path.basename(input_path))[0]
    out_path = os.path.join(out_dir, f"{base}_preprocessed.mp4")
    ff = ffmpeg_exe()

    vf_filters: List[str] = []
    af_filters: List[str] = []

    if deinterlace:
        vf_filters.append("yadif")
    if downscale_720:
        vf_filters.append("scale=-2:720")
    # CFR is not a filter; applied via -r and -vsync later.

    if audio_normalize:
        af_filters.append("loudnorm=I=-16:LRA=11:TP=-1.5")
    if mono_downmix:
        # -ac 1 is simpler; keep af filter empty for downmix and set channels via -ac
        pass
    if trim_silence:
        # Basic removal of leading/trailing silence
        af_filters.append("silenceremove=start_periods=1:start_duration=2:start_threshold=-50dB:stop_periods=1:stop_duration=2:stop_threshold=-50dB")

    vf = ",".join(vf_filters) if vf_filters else "null"
    af = ",".join(af_filters) if af_filters else "anull"

    cmd = [
        ff, "-y",
        "-i", input_path,
        "-vf", vf,
        "-af", af,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
    ]
    if mono_downmix:
        cmd.extend(["-ac", "1"])
    if cfr_30:
        cmd.extend(["-r", "30", "-vsync", "cfr"])
    cmd.append(out_path)

    rc, out, err = run_ffmpeg(cmd)
    if rc != 0 or not os.path.exists(out_path):
        raise RuntimeError(f"FFmpeg preprocessing failed:\n{err}")
    return out_path


def speed_up_15x(input_path: str, out_dir: str) -> str:
    """
    Re-encode to 1.5x speed. Prefer keeping audio with atempo=1.5.
    Fallback to video-only speed-up if audio pipeline fails.
    """
    ensure_dir(out_dir)
    base = os.path.splitext(os.path.basename(input_path))[0]
    out_path = os.path.join(out_dir, f"{base}_x1_5.mp4")
    ff = ffmpeg_exe()

    # Attempt video + audio speed-up
    cmd = [
        ff, "-y",
        "-i", input_path,
        "-vf", "setpts=PTS/1.5",
        "-filter:a", "atempo=1.5",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        out_path,
    ]
    rc, out, err = run_ffmpeg(cmd)
    if rc == 0 and os.path.exists(out_path):
        return out_path

    # Fallback: video-only (no audio)
    out_path_vo = os.path.join(out_dir, f"{base}_x1_5_vo.mp4")
    cmd_vo = [
        ff, "-y",
        "-i", input_path,
        "-vf", "setpts=PTS/1.5",
        "-an",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-movflags", "+faststart",
        out_path_vo,
    ]
    rc2, out2, err2 = run_ffmpeg(cmd_vo)
    if rc2 != 0 or not os.path.exists(out_path_vo):
        raise RuntimeError(f"FFmpeg speed-up failed.\nFirst attempt stderr:\n{err}\nFallback stderr:\n{err2}")
    return out_path_vo


def segment_video(input_path: str, segment_minutes: int, out_dir: str) -> List[str]:
    """
    Split video into N-minute segments with re-encode (H.264 + AAC) for reliability.
    """
    ensure_dir(out_dir)
    seg_dir = os.path.join(out_dir, "segments")
    ensure_dir(seg_dir)
    template = os.path.join(seg_dir, "segment_%03d.mp4")
    seg_seconds = int(segment_minutes * 60)
    ff = ffmpeg_exe()
    cmd = [
        ff, "-y",
        "-i", input_path,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-f", "segment",
        "-segment_time", str(seg_seconds),
        "-reset_timestamps", "1",
        "-movflags", "+faststart",
        template,
    ]
    rc, out, err = run_ffmpeg(cmd)
    if rc != 0:
        raise RuntimeError(f"FFmpeg segmentation failed:\n{err}")

    files = sorted(
        [os.path.join(seg_dir, f) for f in os.listdir(seg_dir) if f.lower().endswith(".mp4")]
    )
    if not files:
        raise RuntimeError("Segmentation produced no files.")
    return files


# --------------- Gemini Files API ---------------

def configure_genai():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("Missing GOOGLE_API_KEY in environment. Create a .env with GOOGLE_API_KEY=your_key")
    genai.configure(api_key=api_key)


def upload_to_files_api(path: str, mime_type: str):
    return genai.upload_file(path=path, mime_type=mime_type)


def poll_until_active(file_obj, timeout_sec: int, poll_sec: float = 2.0):
    deadline = time.time() + timeout_sec
    current = file_obj
    while True:
        state = getattr(getattr(current, "state", None), "name", None)
        if state == "ACTIVE":
            return current
        if state == "FAILED":
            raise RuntimeError("Files API processing failed for uploaded file.")
        if time.time() > deadline:
            raise TimeoutError("Timed out waiting for file to become ACTIVE in Files API.")
        time.sleep(poll_sec)
        current = genai.get_file(current.name)


def delete_files_api(file_obj):
    try:
        genai.delete_file(file_obj.name)
    except Exception:
        pass


def generate_markdown_from_file(
    file_obj,
    mime_type: str,
    prompt_text: str,
    timeout_sec: int,
    transcript_text: Optional[str] = None,
) -> str:
    model = genai.GenerativeModel("gemini-2.5-pro")
    resp = None
    try:
        parts = [genai.Part.from_uri(file_obj.uri, mime_type=mime_type)]
        if transcript_text:
            parts.append({"text": "Transcript context (may contain ASR artifacts):\n" + transcript_text})
        parts.append({"text": prompt_text})
        resp = model.generate_content(parts, request_options={"timeout": timeout_sec})
    except Exception:
        resp = model.generate_content([file_obj, {"text": prompt_text}], request_options={"timeout": timeout_sec})

    text = getattr(resp, "text", None)
    if not text:
        try:
            if getattr(resp, "candidates", None) and resp.candidates and getattr(resp.candidates[0], "content", None):
                pieces = []
                for part in getattr(resp.candidates[0].content, "parts", []) or []:
                    if getattr(part, "text", None):
                        pieces.append(part.text)
                text = "\n".join(pieces).strip()
        except Exception:
            pass
    if not text:
        raise RuntimeError("Empty response from model.")
    return text


# --------------- Prompting ---------------

WATCHLESS_PROMPT = """You are summarizing a video for a reader who MUST NOT have to watch it.
Produce a fully watchless, faithful, lossless reconstruction in Markdown. Do not omit important details.

Output sections:
1) What You’d Know Without Watching
- A comprehensive bullet list of all critical facts, contexts, participants, goals, and outcomes conveyed by the video.

2) Chronological Timeline
- Event-by-event bullets with timestamps [HH:MM:SS] (≈5–15s or event-driven granularity).
- Include: on-screen actions, visible changes, menu or UI interactions, settings/setpoints, parameter changes, readings with units, who is speaking, and short direct quotes in “quotation marks” when they provide rationale or decisions.

3) All Numbers and Measurements (Table)
- Columns: Timestamp | Quantity | Value | Unit | Context
- Include every explicit number (readings, counts, sizes, rates, temperatures, times, etc.) found in the video/audio.

4) Key Decisions and Rationale
- Bullet points citing decisions, their reasons, and any consequences, quoting audio when useful.

5) Issues, Anomalies, Warnings
- Errors, unexpected behaviors, caveats, and mitigations.

6) Action Items / Next Steps
- Concrete follow-ups suggested or implied in the video.

Constraints:
- Prioritize audio for fidelity; include exact terms and units used by the speaker(s).
- No speculation; if something is unknown, use [unclear]/[off‑camera]/[not provided].
- Preserve important wording; include units everywhere applicable.
- The goal is that a busy reader never needs to watch the video to get the full content.
"""

WATCHLESS_LAB_PROMPT = """You are producing a lab‑notebook‑quality, watchless summary of a laboratory video. The reader MUST NOT need to watch it.
Produce an exhaustive, structured Markdown document capturing every essential lab detail.

Output sections:
1) Executive Summary
- Objectives, apparatus, materials, variables, procedures, and outcomes.

2) Step‑by‑Step Timeline (≈5–15 s granularity)
- Chronological bullets with [HH:MM:SS] timestamps.
- Include manipulations (valves/knobs/switches), setpoints/ranges/positions, calibration steps, configuration changes, and short direct quotes for rationale.
- Record all observations: readings with units, colors, flows, temperatures, audible cues, and state changes.

3) Measurements & Readings (Table)
- Columns: Timestamp | Measurement | Value | Unit | Method/Device | Notes

4) Procedures & Settings
- Consolidated list of procedures performed and device settings (setpoints, ranges, modes).

5) Results & Calculations
- Copy numerical results exactly; list formulas or calculations referenced (show units).

6) Anomalies, Errors, and Safety
- Nonconformities, alarms, safety considerations, mitigations.

7) Uncertainty & Repeatability
- Sources of uncertainty, repeatability considerations, calibration notes.

8) Action Items / Next Steps
- Follow‑ups, additional trials, adjustments.

Constraints:
- No speculation; if missing, use [unclear]/[off‑camera]/[not provided].
- Always include units; preserve terminology exactly where possible.
- Treat this like a faithful lab notebook replacement so the reader need not watch the video.
"""

SEGMENT_PREFACE_TEMPLATE = """This summary covers only Segment {index} (approx. {start_hms}–{end_hms}). Focus strictly on this range; do not repeat other segments. Maintain the same structure and constraints."""


def build_prompt(style: str,
                 is_segmented: bool,
                 seg_index: Optional[int] = None,
                 start_s: Optional[int] = None,
                 end_s: Optional[int] = None) -> str:
    if style == "detailed":
        base = WATCHLESS_LAB_PROMPT
    else:
        base = WATCHLESS_PROMPT

    if is_segmented and seg_index is not None and start_s is not None:
        end_hms = "end" if end_s is None else human_hms(end_s)
        preface = SEGMENT_PREFACE_TEMPLATE.format(
            index=seg_index + 1,
            start_hms=human_hms(start_s),
            end_hms=end_hms,
        )
        return preface + "\n\n" + base
    return base


# --------------- YouTube helpers ---------------

YOUTUBE_URL_RE = re.compile(
    r"(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([\w\-]{11})"
)

def parse_youtube_id(url: str) -> Optional[str]:
    m = YOUTUBE_URL_RE.search(url.strip())
    if m:
        return m.group(1)
    return None


def download_youtube(url: str, out_dir: str) -> str:
    if yt_dlp is None:
        raise RuntimeError("yt-dlp not installed. Please install dependencies from requirements.txt.")
    ensure_dir(out_dir)
    ff_dir = os.path.dirname(ffmpeg_exe())

    ydl_opts = {
        "outtmpl": os.path.join(out_dir, "%(title)s.%(ext)s"),
        "format": "bv*+ba/b",  # best video+audio, else best
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = ydl.prepare_filename(info)
    # Ensure mp4 if merged differently
    if not os.path.exists(path) and os.path.exists(path.rsplit(".", 1)[0] + ".mp4"):
        path = path.rsplit(".", 1)[0] + ".mp4"
    if not os.path.exists(path):
        raise RuntimeError("yt-dlp did not produce a file.")
    return path


def fetch_transcript_text(video_id: str, max_chars: int = 15000) -> Optional[str]:
    if YouTubeTranscriptApi is None:
        return None
    try:
        # Try English variants first; fall back to any
        for langs in (["en", "en-US", "en-GB"], []):
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=langs or None)
                text = " ".join([chunk.get("text", "") for chunk in transcript if chunk.get("text")])
                text = re.sub(r"\s+", " ", text).strip()
                if text:
                    return text[:max_chars]
            except (NoTranscriptFound, TranscriptsDisabled):
                continue
        return None
    except (VideoUnavailable, Exception):
        return None


# --------------- Streamlit UI ---------------

st.title("Video Summarizer — Gemini 2.5 Pro (Files API)")
st.caption("Local-first app that uploads to the Gemini Files API (ephemeral) and returns a structured, watchless Markdown summary.")

# Page width (no custom theme)
width_choice = st.selectbox(
    "Page width",
    options=["Narrow (800px)", "Comfort (1000px)", "Wide (1200px)", "Full"],
    index=1
)
width_map = {
    "Narrow (800px)": 800,
    "Comfort (1000px)": 1000,
    "Wide (1200px)": 1200,
    "Full": 1800,
}
inject_width_css(width_map[width_choice])

# Input source
source = st.radio("Input source", options=["Upload file", "YouTube URL"], horizontal=True)

uploaded = None
youtube_url = None
if source == "Upload file":
    uploaded = st.file_uploader(
        "Upload a video",
        type=[e.lstrip(".") for e in ACCEPTED_EXTS],
        accept_multiple_files=False,
        help="Supported: mp4, mov, mkv, webm, avi",
    )
else:
    youtube_url = st.text_input(
        "Paste a YouTube link",
        placeholder="https://www.youtube.com/watch?v=VIDEO_ID or https://youtu.be/VIDEO_ID",
        help="Only use URLs you have rights to process. The app downloads the file temporarily for analysis."
    )

col1, col2 = st.columns([1, 1])
with col1:
    style = st.selectbox(
        "Summary style",
        options=["quick", "detailed"],
        index=0,
        format_func=lambda v: "Watchless (Concise)" if v == "quick" else "Watchless Lab (Detailed)",
        help="Quick: concise but complete. Detailed: lab-notebook style with exhaustive lab details."
    )
with col2:
    delete_after = st.checkbox("Delete from Files API after summarize", value=True)

timeout_min = st.slider("Timeout (minutes)", min_value=2, max_value=30, value=10, step=1)

with st.expander("Preprocessing & Segmentation"):
    st.caption("Optional preprocessing occurs locally before upload. Leave off for fastest results.")
    # Additional preprocessing toggles
    colp1, colp2, colp3 = st.columns(3)
    with colp1:
        downscale_720 = st.checkbox("Downscale to 720p", value=False)
        deinterlace = st.checkbox("Deinterlace (yadif)", value=False)
    with colp2:
        cfr_30 = st.checkbox("Convert to CFR 30fps", value=False)
        audio_normalize = st.checkbox("Normalize audio", value=False)
    with colp3:
        mono_downmix = st.checkbox("Mono downmix", value=False)
        trim_silence = st.checkbox("Trim silence (lead/trail)", value=False)

    speed_up = st.checkbox("Speed up to 1.5×", value=False)
    segmented = st.checkbox("Segmented summarization", value=False)
    segment_len_min = st.slider("Segment length (minutes)", min_value=5, max_value=60, value=15, step=1, disabled=not segmented)
    keep_processed = st.checkbox("Keep processed local files", value=False)

run = st.button("Summarize", type="primary", use_container_width=True)

if run:
    # Basic validations
    if source == "Upload file" and not uploaded:
        st.error("Please upload a video file.")
        st.stop()
    if source == "YouTube URL":
        if not youtube_url:
            st.error("Please paste a YouTube link.")
            st.stop()
        vid = parse_youtube_id(youtube_url)
        if not vid:
            st.error("That does not look like a valid YouTube URL.")
            st.stop()

    try:
        configure_genai()
    except Exception as e:
        st.error(f"Configuration error: {e}")
        st.stop()

    # Save or download into temp
    with tempfile.TemporaryDirectory() as td:
        status = st.status("Starting…", expanded=True)
        transcript_text: Optional[str] = None
        try:
            if source == "Upload file":
                in_name = uploaded.name  # type: ignore
                orig_ext = os.path.splitext(in_name)[1].lower()
                safe_ext = orig_ext if orig_ext in ACCEPTED_EXTS else ".mp4"
                working_path = os.path.join(td, f"input{safe_ext}")
                with open(working_path, "wb") as f:
                    f.write(uploaded.read())  # type: ignore
            else:
                st.write("Downloading YouTube video…")
                ytdir = os.path.join(td, "yt")
                try:
                    working_path = download_youtube(youtube_url, ytdir)  # type: ignore
                    in_name = os.path.basename(working_path)
                    st.success(f"Downloaded: {in_name}")
                except Exception as e:
                    st.error(f"YouTube download failed: {e}")
                    st.stop()
                # Try to fetch transcript to add as context (does not replace video)
                video_id = parse_youtube_id(youtube_url)  # type: ignore
                if video_id:
                    ttxt = fetch_transcript_text(video_id)
                    if ttxt:
                        transcript_text = ttxt
                        st.info("Transcript found and will be used as additional context.")

            processed_paths_to_keep: List[str] = []

            status.update(label="Preprocessing", state="running")
            # Additional preprocessing pass
            try:
                pre_out = preprocess_video(
                    working_path,
                    os.path.join(td, "pre"),
                    downscale_720=downscale_720,
                    cfr_30=cfr_30,
                    audio_normalize=audio_normalize,
                    mono_downmix=mono_downmix,
                    deinterlace=deinterlace,
                    trim_silence=trim_silence,
                )
                if pre_out != working_path:
                    working_path = pre_out
                    processed_paths_to_keep.append(pre_out)
                    st.success("Preprocessing complete.")
            except Exception as e:
                st.warning(f"Preprocessing failed; continuing with original file. Details: {e}")

            if speed_up:
                st.write("Re-encoding at 1.5×…")
                try:
                    spd_dir = os.path.join(td, "speedup")
                    spd_path = speed_up_15x(working_path, spd_dir)
                    working_path = spd_path
                    st.success("Speed-up complete.")
                    processed_paths_to_keep.append(spd_path)
                except Exception as e:
                    st.warning(f"Speed-up failed; continuing with current file. Details: {e}")

            if segmented:
                st.write(f"Segmenting into {segment_len_min}-minute chunks…")
                seg_paths = segment_video(working_path, segment_len_min, os.path.join(td, "segmented"))
                st.success(f"Produced {len(seg_paths)} segment(s).")
                processed_paths_to_keep.extend(seg_paths)
                segments = seg_paths
            else:
                segments = [working_path]

            if keep_processed and processed_paths_to_keep:
                out_root = os.path.join(os.getcwd(), "processed_output", datetime.now().strftime("%Y%m%d_%H%M%S"))
                ensure_dir(out_root)
                for p in processed_paths_to_keep:
                    shutil.copy2(p, os.path.join(out_root, os.path.basename(p)))
                st.info(f"Processed files copied to: {out_root}")

            # Summarization
            total = len(segments)
            prog = st.progress(0.0, text="Summarizing…")
            all_markdowns: List[str] = []

            for idx, seg_path in enumerate(segments):
                st.write(f"Uploading segment {idx+1}/{total} to Files API…")
                mime = mime_from_ext(seg_path)
                file_obj = upload_to_files_api(seg_path, mime_type=mime)
                file_obj = poll_until_active(file_obj, timeout_sec=timeout_min * 60)
                st.success("ACTIVE in Files API.")

                # Build prompt
                if segmented:
                    start_s = idx * segment_len_min * 60
                    end_s = (idx + 1) * segment_len_min * 60
                    prompt = build_prompt(style=style, is_segmented=True, seg_index=idx, start_s=start_s, end_s=end_s)
                else:
                    prompt = build_prompt(style=style, is_segmented=False)

                st.write("Calling Gemini 2.5 Pro…")
                md = generate_markdown_from_file(
                    file_obj,
                    mime,
                    prompt,
                    timeout_sec=timeout_min * 60,
                    transcript_text=transcript_text
                )
                all_markdowns.append(md)
                st.success("Segment summarized.")

                if delete_after:
                    st.write("Deleting file from Files API…")
                    delete_files_api(file_obj)
                    st.success("Deleted.")

                prog.progress((idx + 1) / total, text=f"Summarized {idx+1}/{total}")

            status.update(label="Composing final Markdown", state="running")

            # Compose final output
            if segmented:
                stitched: List[str] = []
                for idx, md in enumerate(all_markdowns):
                    seg_start = human_hms(idx * segment_len_min * 60)
                    seg_end = human_hms((idx + 1) * segment_len_min * 60)
                    header = f"## Segment {idx+1} — {seg_start}–{seg_end}"
                    stitched.append(header + "\n\n" + md.strip())
                final_md = f"# Summary — {in_name}\n\n" + "\n\n---\n\n".join(stitched) + "\n"
            else:
                final_md = f"# Summary — {in_name}\n\n" + all_markdowns[0].strip() + "\n"

            status.update(label="Done", state="complete")
            st.markdown(final_md)

            dl_name = f"{os.path.splitext(in_name)[0]}_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            st.download_button("Download .md", data=final_md.encode("utf-8"), file_name=dl_name, mime="text/markdown")
            st.success("Summary ready.")
        except Exception as e:
            status.update(label="Error", state="error")
            st.error(f"Processing failed: {e}")
