# Video Summarizer — Gemini 2.5 Pro (Files API)

Local-first Streamlit app that generates a fully watchless Markdown summary of videos using Google’s Gemini 2.5 Pro via the Files API. No Drive/OAuth required. Optional preprocessing and segmented summarization for long videos. Temp files cleaned up by default.

## What’s new in this build
- Narrow layout control (centered): Page width presets (Narrow/Comfort/Wide/Full). Default = Comfort (1000px).
- YouTube URL ingestion: Paste a YouTube link and the app downloads the video locally with `yt-dlp` (use only on content you have rights to). It then runs the same Files API pipeline.
- Transcript assist: If available, a YouTube transcript is pulled (via `youtube-transcript-api`) and attached as extra context to improve fidelity (video is still analyzed).
- “Delete from Files API after summarize” is ON by default.
- Extra preprocessing toggles:
  - Downscale to 720p
  - Convert to CFR 30fps
  - Normalize audio (loudnorm)
  - Mono downmix
  - Deinterlace (yadif)
  - Trim silence (lead/trail)
- Existing options kept:
  - 1.5× speed re-encode (with audio atempo fallback to video-only)
  - Segmented summarization for long videos

## Key Features
- Input sources:
  - Upload local file (mp4, mov, mkv, webm, avi)
  - Paste YouTube URL (downloaded locally via `yt-dlp`)
- Gemini Files API:
  - Upload file, poll until ACTIVE, summarize with `gemini-2.5-pro`
  - Optional immediate deletion from Files API after summarization (default: ON)
- Watchless summaries:
  - Quick: faithful, complete “as-if-you-watched-it” output for general videos
  - Detailed (Lab): lab-notebook-grade summary with exhaustive lab details

## Requirements
- Python 3.9+
- A Google API key with access to Gemini 2.5 Pro Files API
- For YouTube URLs: network access; only process content you have rights to

Create a `.env` file in the project root:
```
GOOGLE_API_KEY=your_key_here
```

## Install and Run
Option A: One-click (Windows)
1. Double-click `run_app.bat` (or `run_hidden.vbs` for silent run).
   - Creates `.venv`, installs dependencies, and starts Streamlit.

Option B: Manual
```
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# macOS/Linux (if you just want to run outside Windows launcher):
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Open the app at:
- http://localhost:8501

## UI Controls
- Page width: Narrow | Comfort | Wide | Full
- Input source: Upload file | YouTube URL
- Summary style: `quick` | `detailed`
- Checkbox: Delete from Files API after summarize (default: ON)
- Slider: Timeout (minutes)
- Expander: Preprocessing & Segmentation
  - [ ] Downscale to 720p
  - [ ] Convert to CFR 30fps
  - [ ] Normalize audio
  - [ ] Mono downmix
  - [ ] Deinterlace (yadif)
  - [ ] Trim silence (lead/trail)
  - [ ] Speed up to 1.5×
  - [ ] Segmented summarization
  - Segment length (minutes)
  - [ ] Keep processed local files

## Processing Pipeline
1. Receive upload or download YouTube video with `yt-dlp`.
2. Optional preprocessing in a single FFmpeg pass (as selected).
3. Optional 1.5× re-encode.
4. Optional segmentation into N‑minute chunks.
5. Upload to Files API, poll until ACTIVE.
6. Call `gemini-2.5-pro` with a watchless prompt; optionally attach transcript text as extra context.
7. If segmented, stitch summaries with approximate absolute time ranges.
8. Optional delete files from Files API (default ON).
9. Present Markdown and enable download.
10. Cleanup temp files.

## Notes and Limitations
- Gemini video understanding samples roughly ~1 fps; relies on audio; transcripts help but video is still used.
- Very long videos should use segmented mode to avoid timeouts/token limits.
- YouTube Data API does not provide video downloads and has ToS restrictions. This app uses `yt-dlp` to download for analysis; only use on content you have rights to process.

## Troubleshooting
- Missing GOOGLE_API_KEY: ensure `.env` has `GOOGLE_API_KEY=...`
- yt‑dlp not found: run `pip install -r requirements.txt` (it’s included)
- FFmpeg errors: the app uses `imageio-ffmpeg` portable binary; ensure network/permissions allow execution

## Privacy
- API key stays in `.env` (git-ignored).
- Files API uploads are ephemeral (~48 hours); this app deletes them immediately by default after summarizing.
- Temp files are deleted automatically unless you choose to keep processed outputs.
