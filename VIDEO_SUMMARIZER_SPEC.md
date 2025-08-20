# Video Summarizer — Gemini 2.5 Pro (Files API)

## Overview
Local-first Streamlit app to summarize user-uploaded videos using Google’s Gemini 2.5 Pro via the Files API. No Google Drive required. Videos are uploaded to the Files API (ephemeral ~48 hours), processed by Gemini, and a structured Markdown summary is generated. Optional preprocessing (1.5× speed) and segmented summarization for long videos. Strict temp-file cleanup by default.

## Key Features
- Local file upload only (no Drive/OAuth).
- Uses Google Gemini Files API:
  - Upload video (chunked automatically).
  - Poll file until ACTIVE.
  - Summarize with `gemini-2.5-pro`.
  - Optional immediate delete from Files API after summarization.
- Summary styles:
  - Quick: concise overview + takeaways + timestamps.
  - Detailed (lab-optimized): exhaustive, step-by-step lab record for hands-on physical labs (thermodynamics, fluids, etc.).
- Preprocessing (optional):
  - 1.5× speed-up with FFmpeg (via `imageio-ffmpeg`, portable binary).
  - Preserves audio timing with `atempo=1.5` when available; falls back to video-only if needed.
- Segmented summarization (optional):
  - Split the (original or 1.5×) video into N-minute segments (e.g., 5–30 min).
  - Upload and summarize each segment sequentially.
  - Combine per-segment results into one Markdown with approximate absolute time ranges.
  - Optional delete each segment file from Files API after it’s summarized.
- Strict temp cleanup:
  - All intermediates created in `TemporaryDirectory` and deleted automatically.
  - Optional “Keep processed local files” toggle copies outputs to `./processed_output/`.
- UX:
  - File uploader for: `mp4`, `mov`, `mkv`, `webm`, `avi`.
  - Status messages for each phase: uploading, summarizing, speeding up, segmenting.
  - Timeout slider for model call.
  - Markdown result area + “Download .md”.
- One-click launchers (Windows):
  - `run_app.bat` (auto-create `.venv`, install deps, open app).
  - `run_hidden.vbs` (launch without console window).

## Summary Structure (Detailed/Lab Mode)
- Executive Summary (objectives, apparatus, materials, variables, procedures, outcomes).
- Step-by-Step Timeline with ~5–15 s granularity:
  - Manipulations (valves/knobs/switches), setpoints, calibration.
  - Readings (with units), observations (color/flow/temp), rationale (quoted audio), state changes.
- Measurements & Readings (consolidated table).
- Procedures & Settings (setpoints/ranges/positions).
- Results & Calculations (copy exactly; list formulas referenced).
- Anomalies, Errors, and Safety.
- Uncertainty & Repeatability.
- Action Items / Next Steps.
- Constraints: no speculation; use `[unclear]`/`[off‑camera]`/`[not provided]`; always include units.
- Notes FPS: Gemini video understanding samples ≈1 fps and uses audio; rely on audio for fidelity.

## UI Controls
- Summary style: `quick` | `detailed`.
- Checkbox: Delete from Files API after summarize.
- Slider: Timeout (min).
- Expander: Preprocessing & Segmentation:
  - [ ] Speed up to 1.5×.
  - [ ] Segmented summarization.
  - Segment length (minutes).
  - [ ] Keep processed local files.

## Processing Pipeline
1) Receive user upload.
2) Optionally re-encode at 1.5× (H.264 + AAC + `+faststart`).
3) If segmented, split into N-minute chunks; otherwise proceed single file.
4) Upload to Files API, poll until ACTIVE.
5) Call `gemini-2.5-pro` with file reference and structured prompt.
6) Collect response text.
7) If segmented, stitch all segment summaries with absolute time ranges.
8) Optionally delete Files API files.
9) Present Markdown + download button.
10) Cleanup all temp files by default.

## Tech Stack
- Python, Streamlit.
- `google-generativeai` (Files API + GenerativeModel).
- `imageio-ffmpeg` (portable FFmpeg).
- `python-dotenv`.

## Configuration
- `.env`: `GOOGLE_API_KEY=your_key_here`.
- `.gitignore` blocks `.env` and `.streamlit`.
- Windows scripts: `run_app.bat` (creates `.venv`, installs, launches), `run_hidden.vbs`.

## Commands (local dev)
- `pip install -r requirements.txt`
- `streamlit run app.py`
- Or double-click `run_app.bat`.

## Dependencies
- `streamlit`
- `google-generativeai`
- `python-dotenv`
- `imageio-ffmpeg`

## Security & Privacy
- Key in `.env` (ignored by git).
- Files API uploads are ephemeral (~48 hours).
- Optional immediate deletion post-summary.
- Temp files deleted by default unless user opts to keep.

## Limitations & Notes
- ≈1 fps frame sampling by model; audio is crucial.
- Extremely long videos may need segmented mode to avoid timeouts.
- Output length limited by model tokens; segmentation mitigates this.

## Future Enhancements (Optional)
- “Transcribe first” path (ASR) + text-based summarization.
- Scene-change-based segmentation.
- Domain presets (e.g., chemistry/mechatronics).
- Highlight extraction (alarms/errors/threshold breaches).
- Save summaries to Drive or other storage providers.
- Batch queue for multiple videos.
