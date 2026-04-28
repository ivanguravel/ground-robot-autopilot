"""
Gradio Web UI for Autopilot.

Upload video or paste YouTube URL → Process with autopilot → View result + Chat about decisions.

Usage:
    python app.py
    python app.py --llm-provider ollama
    python app.py --llm-provider azure
    
Then open http://localhost:7860 in browser.
"""

import os
import json
import time
import re
import argparse
import threading
import tempfile
import gradio as gr
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from openai import AzureOpenAI, OpenAI

# Parse --llm-provider before other imports that might use it
_parser = argparse.ArgumentParser()
_parser.add_argument(
    "--llm-provider",
    choices=["azure", "ollama"],
    default="azure",
    help="LLM provider for autopilot and chat (default: azure)",
)
_args, _ = _parser.parse_known_args()
LLM_PROVIDER_APP = _args.llm_provider

# YouTube download support
try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False
    print("⚠️ yt-dlp not installed. Run: pip install yt-dlp")

try:
    from pytubefix import YouTube as PTFYouTube
    PYTUBEFIX_AVAILABLE = True
except ImportError:
    PYTUBEFIX_AVAILABLE = False

from config import DEVICE

try:
    from config import (
        AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY,
        AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_API_VERSION,
    )
except ImportError:
    AZURE_OPENAI_ENDPOINT = ""
    AZURE_OPENAI_API_KEY = ""
    AZURE_OPENAI_DEPLOYMENT = "gpt-4o-mini"
    AZURE_OPENAI_API_VERSION = "2024-12-01-preview"

try:
    from config import OLLAMA_BASE_URL, OLLAMA_MODEL
except ImportError:
    OLLAMA_BASE_URL = "http://localhost:11434/v1"
    OLLAMA_MODEL = "llama3.2:3b"
# CPU optimization: fewer frames + less frequent LLM calls
_IS_CPU = str(DEVICE) == "cpu"
DEFAULT_SKIP_FRAMES = 25 if _IS_CPU else 10
DEFAULT_LLM_INTERVAL = 2.5 if _IS_CPU else 1.5


# ============================================
# Global State for Background Processing
# ============================================
class ProcessingState:
    def __init__(self):
        self.is_processing = False
        self.current_frame = None
        self.status = "Ready"
        self.progress = 0
        self.entries_table = ""
        self.log_file = None
        self.lock = threading.Lock()
    
    def update(self, frame=None, status=None, progress=None, entries=None, log_file=None):
        with self.lock:
            if frame is not None:
                self.current_frame = frame
            if status is not None:
                self.status = status
            if progress is not None:
                self.progress = progress
            if entries is not None:
                self.entries_table = entries
            if log_file is not None:
                self.log_file = log_file
    
    def get(self):
        with self.lock:
            return self.current_frame, self.status, self.progress, self.entries_table

state = ProcessingState()


# ============================================
# YouTube Download
# ============================================

def is_youtube_url(url: str) -> bool:
    """Check if string is a YouTube URL."""
    if not url:
        return False
    youtube_patterns = [
        r'(youtube\.com/watch\?v=)',
        r'(youtu\.be/)',
        r'(youtube\.com/embed/)',
        r'(youtube\.com/v/)',
        r'(youtube\.com/shorts/)',
    ]
    return any(re.search(pattern, url) for pattern in youtube_patterns)


COOKIES_PATH = os.path.join(os.path.dirname(__file__), "cookies.txt")


def _download_via_ytdlp(url: str, max_duration: int, download_dir: str) -> Tuple[Optional[str], Optional[str]]:
    """Try downloading via yt-dlp. Returns (path, error_or_None)."""
    if not YT_DLP_AVAILABLE:
        return None, "yt-dlp not available"

    yt_ver = getattr(yt_dlp.version, '__version__', 'unknown')
    print(f"[YouTube/yt-dlp] version {yt_ver}")

    def progress_hook(d):
        if d['status'] == 'downloading':
            percent = d.get('_percent_str', '?%').strip()
            speed = d.get('_speed_str', '?').strip()
            state.update(status=f"⬇️ Downloading: {percent} @ {speed}")
        elif d['status'] == 'finished':
            state.update(status="✅ Download complete, processing...")

    base_opts = {
        'outtmpl': os.path.join(download_dir, '%(id)s.%(ext)s'),
        'progress_hooks': [progress_hook],
        'js_runtimes': {'node': {}, 'deno': {}, 'bun': {}},
    }
    if os.path.exists(COOKIES_PATH):
        base_opts['cookiefile'] = COOKIES_PATH

    strategies = [
        ("cookies + default client", {}),
        ("cookies + android client", {'extractor_args': {'youtube': {'player_client': ['android']}}}),
        ("cookies + ios client", {'extractor_args': {'youtube': {'player_client': ['ios']}}}),
        ("no cookies", {'cookiefile': None}),
    ]
    if not os.path.exists(COOKIES_PATH):
        strategies = [("no cookies", {})]

    last_error = ""
    for label, extra in strategies:
        try:
            opts = {**base_opts, **extra}
            if 'cookiefile' in extra and extra['cookiefile'] is None:
                opts.pop('cookiefile', None)

            state.update(status=f"🔍 yt-dlp: {label}...")
            print(f"[YouTube/yt-dlp] Strategy: {label}")

            with yt_dlp.YoutubeDL({**opts, 'quiet': True}) as ydl:
                info = ydl.extract_info(url, download=False, process=False)

            title = info.get('title', 'Unknown')
            duration = info.get('duration') or 0
            formats = info.get('formats') or []
            print(f"[YouTube/yt-dlp] Title: {title}, Duration: {duration}s, Formats: {len(formats)}")

            if not formats:
                print(f"[YouTube/yt-dlp] No formats with '{label}', trying next...")
                continue

            dl_opts = {**opts, 'quiet': True, 'no_warnings': True}
            dl_opts['format'] = 'bestvideo[height<=720]+bestaudio/best[height<=720]/bestvideo+bestaudio/best'
            dl_opts['merge_output_format'] = 'mp4'

            if duration and duration > max_duration:
                print(f"[YouTube/yt-dlp] Trimming to first {max_duration}s")
                state.update(status=f"⬇️ Downloading first {max_duration}s...")
                dl_opts['download_ranges'] = yt_dlp.utils.download_range_func(None, [(0, max_duration)])
                dl_opts['force_keyframes_at_cuts'] = True
            else:
                state.update(status=f"⬇️ Downloading: {title[:30]}...")

            with yt_dlp.YoutubeDL(dl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                video_path = ydl.prepare_filename(info)

            if not os.path.exists(video_path):
                video_id = info.get('id', '')
                for ext in ['mp4', 'webm', 'mkv']:
                    alt = os.path.join(download_dir, f"{video_id}.{ext}")
                    if os.path.exists(alt):
                        video_path = alt
                        break

            if os.path.exists(video_path):
                size_mb = os.path.getsize(video_path) / (1024 * 1024)
                print(f"[YouTube/yt-dlp] OK: {video_path} ({size_mb:.1f} MB)")
                return video_path, None

        except Exception as e:
            last_error = str(e)
            print(f"[YouTube/yt-dlp] '{label}' failed: {last_error[:120]}")
            if "Video unavailable" in last_error:
                return None, "Video unavailable or private"
            continue

    return None, last_error or "All yt-dlp strategies failed"


def _download_via_pytubefix(url: str, max_duration: int, download_dir: str) -> Tuple[Optional[str], Optional[str]]:
    """Try downloading via pytubefix with automatic PO token (Node.js). Returns (path, error_or_None)."""
    if not PYTUBEFIX_AVAILABLE:
        return None, "pytubefix not installed"

    state.update(status="🔍 pytubefix: auto PO token via Node.js...")
    print("[YouTube/pytubefix] Trying with WEB client (auto PO token)...")

    clients_to_try = ['WEB', 'WEB_CREATOR', 'ANDROID', 'IOS']
    last_error = ""

    for client in clients_to_try:
        try:
            print(f"[YouTube/pytubefix] Client: {client}")
            state.update(status=f"🔍 pytubefix: {client} client...")

            yt = PTFYouTube(url, client)

            title = yt.title or "Unknown"
            duration = yt.length or 0
            print(f"[YouTube/pytubefix] Title: {title}, Duration: {duration}s")

            if duration and duration > max_duration:
                print(f"[YouTube/pytubefix] Video is {duration}s, will trim to {max_duration}s after download")

            state.update(status=f"⬇️ pytubefix: downloading {title[:30]}...")

            stream = (
                yt.streams
                .filter(progressive=True, file_extension='mp4')
                .order_by('resolution')
                .desc()
                .first()
            )
            if not stream:
                stream = yt.streams.filter(file_extension='mp4').order_by('resolution').desc().first()
            if not stream:
                stream = yt.streams.get_highest_resolution()
            if not stream:
                print(f"[YouTube/pytubefix] No streams for client {client}")
                last_error = "No streams available"
                continue

            print(f"[YouTube/pytubefix] Stream: {stream.resolution}, {stream.mime_type}")
            out_path = stream.download(output_path=download_dir)

            if out_path and os.path.exists(out_path):
                if duration and duration > max_duration:
                    trimmed = _trim_video(out_path, max_duration, download_dir)
                    if trimmed:
                        out_path = trimmed

                size_mb = os.path.getsize(out_path) / (1024 * 1024)
                print(f"[YouTube/pytubefix] OK: {out_path} ({size_mb:.1f} MB)")
                return out_path, None

        except Exception as e:
            last_error = str(e)
            print(f"[YouTube/pytubefix] Client '{client}' failed: {last_error[:150]}")
            continue

    return None, last_error or "All pytubefix clients failed"


def _trim_video(video_path: str, max_seconds: int, output_dir: str) -> Optional[str]:
    """Trim video to max_seconds using ffmpeg. Returns trimmed path or None."""
    import shutil
    if not shutil.which('ffmpeg'):
        print("[YouTube] ffmpeg not found, skipping trim")
        return None
    try:
        import subprocess
        base = os.path.splitext(os.path.basename(video_path))[0]
        trimmed_path = os.path.join(output_dir, f"{base}_trimmed.mp4")
        cmd = [
            'ffmpeg', '-y', '-i', video_path,
            '-t', str(max_seconds), '-c', 'copy', trimmed_path
        ]
        subprocess.run(cmd, capture_output=True, timeout=120)
        if os.path.exists(trimmed_path) and os.path.getsize(trimmed_path) > 0:
            os.remove(video_path)
            print(f"[YouTube] Trimmed to {max_seconds}s: {trimmed_path}")
            return trimmed_path
    except Exception as e:
        print(f"[YouTube] Trim failed: {e}")
    return None


def download_youtube_video(url: str, max_duration: int = 300) -> Tuple[Optional[str], str]:
    """
    Download YouTube video to temp file.
    Strategy: yt-dlp first, then pytubefix (auto PO token) as fallback.
    """
    download_dir = os.path.join(tempfile.gettempdir(), "autopilot_youtube")
    os.makedirs(download_dir, exist_ok=True)

    # --- Attempt 1: yt-dlp ---
    path, err = _download_via_ytdlp(url, max_duration, download_dir)
    if path:
        size_mb = os.path.getsize(path) / (1024 * 1024)
        title = os.path.splitext(os.path.basename(path))[0]
        return path, f"✅ Downloaded: {title[:40]} ({size_mb:.1f} MB)"

    ytdlp_err = err or "unknown error"
    bot_blocked = any(k in ytdlp_err.lower() for k in ["sign in", "bot", "confirm"])
    print(f"[YouTube] yt-dlp failed{' (bot detection)' if bot_blocked else ''}: {ytdlp_err[:100]}")

    # --- Attempt 2: pytubefix with auto PO token ---
    if PYTUBEFIX_AVAILABLE:
        print("[YouTube] Falling back to pytubefix...")
        state.update(status="🔄 Switching to pytubefix (auto PO token)...")
        path, err = _download_via_pytubefix(url, max_duration, download_dir)
        if path:
            size_mb = os.path.getsize(path) / (1024 * 1024)
            title = os.path.splitext(os.path.basename(path))[0]
            return path, f"✅ Downloaded via pytubefix: {title[:40]} ({size_mb:.1f} MB)"
        ptf_err = err or "unknown error"
        print(f"[YouTube] pytubefix also failed: {ptf_err[:100]}")
    else:
        ptf_err = "not installed (pip install pytubefix)"

    yt_ver = getattr(yt_dlp.version, '__version__', 'unknown') if YT_DLP_AVAILABLE else 'N/A'
    return None, (
        "❌ All download methods failed.\n"
        f"  yt-dlp ({yt_ver}): {ytdlp_err[:80]}\n"
        f"  pytubefix: {ptf_err[:80]}\n\n"
        "Try:\n"
        "1. pip install -U pytubefix (uses auto PO token via Node.js)\n"
        "2. Different video URL\n"
        "3. Re-export cookies.txt for yt-dlp"
    )


# ============================================
# Autopilot Processing (background thread)
# ============================================

def process_video_background(video_path: str, skip_frames: int, detection_threshold: float):
    """Process video in background thread, updating global state."""
    from autopilot import Autopilot
    
    # Note: state.is_processing is set by caller (process_video_with_download or directly)
    if not state.is_processing:
        state.is_processing = True
    state.update(status="⏳ Initializing autopilot...", progress=0)
    
    try:
        import config as _cfg
        _cfg.LLM_PROVIDER = LLM_PROVIDER_APP

        import inspect
        sig_params = set(inspect.signature(Autopilot.__init__).parameters)
        want = dict(
            detection_threshold=detection_threshold,
            llm_interval=DEFAULT_LLM_INTERVAL,
            use_llm=True,
            debug_llm=False,
            llm_provider=LLM_PROVIDER_APP,
        )
        init_kwargs = {k: v for k, v in want.items() if k in sig_params}
        autopilot = Autopilot(**init_kwargs)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            state.update(status="❌ Error: Cannot open video", progress=0)
            state.is_processing = False
            return
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        state.update(log_file=autopilot.log_file)
        
        frame_count = 0
        start_time = time.time()
        
        while state.is_processing:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Frame skipping
            if skip_frames > 1 and frame_count % skip_frames != 0:
                frame_count += 1
                continue
            
            # Compute movement
            autopilot.compute_movement_vector(frame)
            
            # Detection
            boxes, labels, scores = autopilot.detect_objects(frame)
            
            # Get command
            command = autopilot.get_control_command(
                boxes, labels, scores, width, height, frame_num=frame_count
            )
            
            # FPS
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            autopilot._last_fps = fps
            
            # Draw overlay
            result_frame = autopilot.draw_overlay(frame, boxes, labels, scores, command, fps)
            
            frame_count += 1
            
            # Update state
            pct = int(frame_count / total_frames * 100)
            status = f"🎬 {frame_count}/{total_frames} ({pct}%) | FPS: {fps:.1f}"
            frame_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
            state.update(frame=frame_rgb, status=status, progress=pct)
        
        cap.release()
        
        # Load log for chat
        if state.log_file:
            chat.load_log(state.log_file)
        
        total_time = time.time() - start_time
        final_status = f"✅ Done! {frame_count} frames in {total_time:.1f}s"
        state.update(status=final_status, progress=100, entries=chat.get_entries_table())
        
    except Exception as e:
        state.update(status=f"❌ Error: {e}", progress=0)
    
    state.is_processing = False


def start_processing(video_path, youtube_url, skip_frames, threshold, max_duration):
    """Start background processing from uploaded video or YouTube URL."""
    
    if state.is_processing:
        return "⚠️ Already processing..."
    
    use_youtube = youtube_url and youtube_url.strip() and is_youtube_url(youtube_url.strip())
    
    if not use_youtube and video_path is None:
        return "❌ Upload a video or paste YouTube URL"
    
    if use_youtube and not YT_DLP_AVAILABLE:
        return "❌ yt-dlp not installed. Run: pip install yt-dlp"
    
    thread = threading.Thread(
        target=process_video_with_download,
        args=(video_path, youtube_url.strip() if use_youtube else None,
              skip_frames, threshold, int(max_duration)),
        daemon=True
    )
    thread.start()
    
    if use_youtube:
        return "⬇️ Downloading from YouTube..."
    return "🚀 Processing started..."


def process_video_with_download(video_path, youtube_url, skip_frames, threshold, max_duration):
    """Background processing with optional YouTube download."""
    state.is_processing = True
    
    final_video_path = video_path
    
    if youtube_url:
        final_video_path, status = download_youtube_video(youtube_url, max_duration=max_duration)
        if final_video_path is None:
            state.update(status=status, progress=0)
            state.is_processing = False
            return
    
    process_video_background(final_video_path, skip_frames, threshold)


def stop_processing():
    """Stop background processing and reset state."""
    state.is_processing = False
    state.update(status="Ready", progress=0)
    return "⏹️ Stopped"


def get_current_state():
    """Get current processing state for UI update."""
    frame, status, progress, entries = state.get()
    return frame, status, progress, entries


# ============================================
# Chat System (from chat_autopilot.py)
# ============================================

class GradioChat:
    """Chat system for Gradio interface."""
    
    def __init__(self, provider: str = "azure"):
        self.entries: List[Dict] = []
        self.log_file: str = None
        self.current_entry: Optional[Dict] = None
        self.conversation_history: List[Dict] = []
        
        # Initialize LLM client based on provider
        try:
            if provider == "ollama":
                self.client = OpenAI(
                    base_url=OLLAMA_BASE_URL,
                    api_key="ollama",
                )
                self.deployment = OLLAMA_MODEL
                print(f"Chat LLM: Ollama ({OLLAMA_MODEL})")
            else:
                self.client = AzureOpenAI(
                    azure_endpoint=AZURE_OPENAI_ENDPOINT,
                    api_key=AZURE_OPENAI_API_KEY,
                    api_version=AZURE_OPENAI_API_VERSION,
                )
                self.deployment = AZURE_OPENAI_DEPLOYMENT
                print(f"Chat LLM: Azure ({AZURE_OPENAI_DEPLOYMENT})")
        except Exception as e:
            print(f"LLM init error: {e}")
            self.client = None
    
    def load_log(self, log_file: str) -> str:
        """Load log file."""
        self.entries = []
        self.log_file = log_file
        self.current_entry = None
        self.conversation_history = []
        
        if not log_file or not os.path.exists(log_file):
            return "❌ Log file not found"
        
        with open(log_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        self.entries.append(entry)
                    except json.JSONDecodeError:
                        continue
        
        return f"✅ Loaded {len(self.entries)} entries"
    
    def get_entries_table(self) -> str:
        """Get entries as markdown table."""
        if not self.entries:
            return "No entries loaded. Process a video first."
        
        lines = ["| # | Frame | Objects | Pedal | Gear | Reasoning |",
                 "|---|-------|---------|-------|------|-----------|"]
        
        for i, entry in enumerate(self.entries[:50]):  # Limit to 50
            frame = entry.get('frame', 'N/A')
            objects = len(entry.get('detected_objects', []))
            cmd = entry.get('command', {})
            pedal = f"{cmd.get('pedal', '?')} {cmd.get('pedal_percent', 0)}%"
            gear = cmd.get('gear', '?')
            reasoning = cmd.get('reasoning', '')[:30]
            
            lines.append(f"| {i} | {frame} | {objects} | {pedal} | {gear} | {reasoning}... |")
        
        if len(self.entries) > 50:
            lines.append(f"\n*...showing 50 of {len(self.entries)} entries*")
        
        return "\n".join(lines)
    
    def select_entry(self, index: int) -> str:
        """Select entry by index."""
        if not self.entries:
            return "No entries loaded."
        
        if 0 <= index < len(self.entries):
            self.current_entry = self.entries[index]
            self.conversation_history = []
            return self._format_entry()
        else:
            return f"Invalid index. Valid: 0-{len(self.entries)-1}"
    
    def _format_entry(self) -> str:
        """Format current entry for display."""
        if not self.current_entry:
            return "No entry selected"
        
        entry = self.current_entry
        
        lines = [
            f"## 📍 Entry {self.entries.index(entry)}",
            f"**Frame:** {entry.get('frame', 'N/A')}",
            f"**Timestamp:** {entry.get('timestamp', 'N/A')}",
            "",
            "### 🎯 Detected Objects:"
        ]
        
        objects = entry.get('detected_objects', [])
        if objects:
            for obj in objects:
                if isinstance(obj, dict):
                    name = obj.get('class_name', obj.get('name', '?'))
                    pos = obj.get('position', '?')
                    dist_m = obj.get('distance_meters', obj.get('distance_m', '?'))
                    lines.append(f"- **{name}**: position={pos}, distance={dist_m}m")
        else:
            lines.append("- (no objects)")
        
        cmd = entry.get('command', {})
        lines.extend([
            "",
            "### 🎮 Decision:",
            f"- **Steering:** {cmd.get('steering', '?')} ({cmd.get('steering_angle', 0)}°)",
            f"- **Pedal:** {cmd.get('pedal', '?')} @ {cmd.get('pedal_percent', 0)}%",
            f"- **Gear:** {cmd.get('gear', '?')}",
            f"- **Reasoning:** {cmd.get('reasoning', 'N/A')}"
        ])
        
        return "\n".join(lines)
    
    def ask(self, question: str) -> str:
        """Ask question about current entry."""
        if not self.client:
            return "❌ LLM not initialized"
        
        if not self.current_entry:
            return "❌ Select an entry first (enter a number in the Entry Index field)"
        
        # Build context
        entry = self.current_entry
        objects = entry.get('detected_objects', [])
        
        obj_lines = []
        for obj in objects:
            if isinstance(obj, dict):
                name = obj.get('class_name', '?')
                pos = obj.get('position', '?')
                dist_m = obj.get('distance_meters', '?')
                obj_lines.append(f"- {name}: position={pos}, distance={dist_m}m")
        
        objects_text = "\n".join(obj_lines) if obj_lines else "No objects detected"
        cmd = entry.get('command', {})
        
        context = f"""
SITUATION AT FRAME {entry.get('frame', 'N/A')}:

Detected objects:
{objects_text}

Decision made:
- Steering: {cmd.get('steering', '?')} at {cmd.get('steering_angle', 0)}°
- Pedal: {cmd.get('pedal', '?')} at {cmd.get('pedal_percent', 0)}%
- Gear: {cmd.get('gear', '?')}
- Reasoning: {cmd.get('reasoning', 'N/A')}
"""
        
        system_prompt = """You are an AI explaining autopilot decisions for a toy car.
Explain WHY decisions were made based on detected objects, positions, distances.
Be concise. Speak in first person as if you made the decision."""
        
        # Build conversation
        if not self.conversation_history:
            self.conversation_history.append({
                "role": "user",
                "content": f"Here is the situation:\n{context}"
            })
            self.conversation_history.append({
                "role": "assistant",
                "content": "I see. What would you like to know about this decision?"
            })
        
        self.conversation_history.append({"role": "user", "content": question})
        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    *self.conversation_history
                ],
                temperature=0.7,
                max_tokens=500,
            )
            
            answer = response.choices[0].message.content
            self.conversation_history.append({"role": "assistant", "content": answer})
            
            return answer
            
        except Exception as e:
            return f"❌ Error: {e}"


# ============================================
# Gradio Interface
# ============================================

# Global chat instance (uses LLM_PROVIDER_APP from --llm-provider arg)
chat = GradioChat(provider=LLM_PROVIDER_APP)


def process_and_show(video, youtube_url, skip_frames, threshold, max_duration):
    """Start processing and return initial status."""
    return start_processing(video, youtube_url, skip_frames, threshold, max_duration)


def select_entry_handler(index):
    """Handle entry selection."""
    try:
        idx = int(index)
        details = chat.select_entry(idx)
        return details
    except:
        return "Enter a valid entry number"


def chat_handler(message, history):
    """Handle chat messages."""
    if history is None:
        history = []
    
    if not message or not message.strip():
        return history
    
    try:
        # Reload log from file if processing (log updates in real-time)
        if state.log_file and os.path.exists(state.log_file):
            chat.load_log(state.log_file)
        
        # Check if it's a number (entry selection)
        if message.strip().isdigit():
            idx = int(message.strip())
            if chat.entries:
                if idx >= len(chat.entries):
                    idx = len(chat.entries) - 1
                chat.select_entry(idx)
                entry = chat.current_entry
                cmd = entry.get('command', {})
                response = f"📍 Entry {idx}: Frame {entry.get('frame', '?')} | {cmd.get('pedal', '?')} {cmd.get('pedal_percent', 0)}% | Gear {cmd.get('gear', '?')}\n\nAsk me anything about this decision!"
            else:
                response = "⚠️ No entries yet. Wait for some frames to process..."
        else:
            # Ask question - auto-select last entry if none selected
            if not chat.entries:
                response = "⚠️ No entries yet. Wait for some frames to process..."
            else:
                if not chat.current_entry:
                    # Auto-select last entry
                    chat.select_entry(len(chat.entries) - 1)
                    response = f"📍 Auto-selected latest entry ({len(chat.entries)-1}).\n\n"
                    response += chat.ask(message)
                else:
                    response = chat.ask(message)
    except Exception as e:
        response = f"❌ Error: {str(e)}"
    
    # Append to history
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    return history


# Build Gradio interface
with gr.Blocks(title="🚗 Autopilot Dashboard") as demo:
    
    gr.Markdown("# 🚗 Autopilot Dashboard")
    gr.Markdown(f"Upload video **or** paste YouTube URL | LLM: **{LLM_PROVIDER_APP}**")
    
    # Top row: Upload & YouTube URL
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Tab("📹 Upload"):
                input_video = gr.Video(label="Upload Video", sources=["upload"])
            with gr.Tab("🔗 YouTube"):
                youtube_url = gr.Textbox(
                    label="YouTube URL",
                    placeholder="https://www.youtube.com/watch?v=... or https://youtu.be/...",
                    info="Paste any YouTube video URL (max 5 min by default)"
                )
                _has_cookies = os.path.exists(COOKIES_PATH)
                _ptf = "✅ pytubefix" if PYTUBEFIX_AVAILABLE else "❌ pytubefix (pip install pytubefix)"
                _cookies_msg = f"✅ cookies.txt found" if _has_cookies else "⚠️ No cookies.txt"
                gr.Markdown(
                    f"**Downloaders:** yt-dlp → pytubefix (fallback)  \n"
                    f"{_cookies_msg} | {_ptf}"
                )
        
        with gr.Column(scale=1):
            skip_frames = gr.Slider(minimum=1, maximum=30, value=DEFAULT_SKIP_FRAMES, step=1, label="Skip Frames", info="Higher = faster (CPU: use 15+)")
            threshold = gr.Slider(minimum=0.3, maximum=0.9, value=0.5, step=0.05, label="Detection Threshold")
            max_duration = gr.Slider(minimum=60, maximum=900, value=300, step=60, label="Max Duration (sec)", info="Long videos will be trimmed automatically")
            with gr.Row():
                process_btn = gr.Button("🚀 Process", variant="primary")
                stop_btn = gr.Button("⏹️ Stop", variant="secondary")
    
    # Status bar
    with gr.Row():
        status_text = gr.Textbox(label="Status", value="Ready", interactive=False, scale=4)
        progress_bar = gr.Slider(minimum=0, maximum=100, value=0, label="%", interactive=False, scale=1)
    
    # Main content: Live Preview + Chat side by side
    with gr.Row():
        # Left: Live video preview
        with gr.Column(scale=3):
            gr.Markdown("### 🔴 Live Processing (updates every 0.5s)")
            live_preview = gr.Image(label="Autopilot View", height=450)
        
        # Right: Chat (works during processing!)
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Chat (available during processing!)")
            chatbot = gr.Chatbot(height=350)
            
            with gr.Row():
                entry_index = gr.Number(label="Entry #", precision=0, scale=1)
                chat_input = gr.Textbox(label="Question", placeholder="Type question...", scale=3)
                chat_btn = gr.Button("Send", variant="secondary", scale=1)
            
            entry_details = gr.Markdown("*Select entry to see details*")
    
    # Bottom: Decision log (collapsible)
    with gr.Accordion("📊 Decision Log (click to expand)", open=False):
        entries_table = gr.Markdown("Process a video to see entries")
    
    # Timer for updating UI (every 0.5 sec)
    timer = gr.Timer(0.5, active=True)
    
    # Event handlers
    process_btn.click(
        fn=process_and_show,
        inputs=[input_video, youtube_url, skip_frames, threshold, max_duration],
        outputs=[status_text]
    )
    
    stop_btn.click(
        fn=stop_processing,
        outputs=[status_text]
    )
    
    # Periodic UI update
    timer.tick(
        fn=get_current_state,
        outputs=[live_preview, status_text, progress_bar, entries_table]
    )
    
    entry_index.change(
        fn=select_entry_handler,
        inputs=[entry_index],
        outputs=[entry_details]
    )
    
    chat_btn.click(
        fn=chat_handler,
        inputs=[chat_input, chatbot],
        outputs=[chatbot]
    ).then(
        fn=lambda: "",
        outputs=[chat_input]
    )
    
    chat_input.submit(
        fn=chat_handler,
        inputs=[chat_input, chatbot],
        outputs=[chatbot]
    ).then(
        fn=lambda: "",
        outputs=[chat_input]
    )
    
    # Auto-stop processing when user refreshes or closes the page
    demo.unload(fn=stop_processing)
    
    gr.Markdown(f"""
    ---
    **Tips:**
    - 📹 Upload a video file **or** paste a YouTube URL
    - 💬 Chat works DURING processing! Enter entry # then ask questions
    - ⚡ CPU mode: Skip Frames 7+ recommended. Ollama: try `phi3:mini` or `qwen2.5:1.5b` for speed
    """)


if __name__ == "__main__":
    print(f"🚀 Starting with LLM provider: {LLM_PROVIDER_APP}")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate")
    )
