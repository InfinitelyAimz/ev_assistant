import os
import sys
import json
import re
import html
import urllib.parse
import platform
import webbrowser
import subprocess
import time
import datetime
import ctypes
import math
import random
import warnings
import socket
from collections import deque

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import sounddevice as sd
import numpy as np
from scipy.io import wavfile
from faster_whisper import WhisperModel
import ollama
import psutil
import requests
import screen_brightness_control as sbc
from pycaw.pycaw import AudioUtilities

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QHBoxLayout, QLabel, QFrame, QPushButton, QTextEdit, QLineEdit
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QPoint, QPointF
from PyQt6.QtGui import (
    QColor, QFont, QPainter, QBrush, QPen, QRadialGradient, 
    QLinearGradient, QPainterPath, QKeyEvent, QTextCursor
)

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

try:
    from ytmusicapi import YTMusic
    ytmusic_client = YTMusic()
except Exception:
    ytmusic_client = None

CURRENT_MIC_RMS = 0.0
START_TIME = psutil.boot_time()
LAST_NET_IO = psutil.net_io_counters()
LAST_NET_TIME = time.time()
NET_UP_SPEED = 0.0
NET_DOWN_SPEED = 0.0

IS_USER_TYPING = False
LAST_TYPING_TIME = 0.0

# -------------------------------------------------------------
# 1. VOICE ENGINE (LIGHTNING-FAST OFFLINE PIPER + ULTRON DSP)
# -------------------------------------------------------------
piper_dir = os.path.join(BASE_DIR, "piper")
piper_exe = os.path.join(piper_dir, "piper.exe")
voice_model = os.path.join(piper_dir, "jarvis-medium.onnx")
json_config = os.path.join(piper_dir, "jarvis-medium.onnx.json")
temp_wav_path = os.path.join(BASE_DIR, "temp_input.wav")

def speak(text, worker_ref=None, force_speech=False):
    """Synthesizes speech using local Piper engine with a smooth, slightly higher-pitched humanized rasp."""
    global CURRENT_MIC_RMS
    if not text or not text.strip():
        return

    clean_text = text.strip()
    words = clean_text.split()
    print(f"\n[E.V. (Smooth Humanized Ultron)]: {clean_text}")

    is_silent = worker_ref and worker_ref.voice_silent_mode and not force_speech
    if is_silent:
        if worker_ref:
            worker_ref.stream_start.emit()
            for w in words:
                worker_ref.stream_word.emit(w + " ")
                time.sleep(0.035)
            worker_ref.stream_end.emit()
        return

    if os.path.exists(piper_exe) and os.path.exists(voice_model):
        try:
            process = subprocess.Popen(
                [piper_exe, "-m", voice_model, "-c", json_config, "--output_raw"],
                cwd=piper_dir,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL
            )
            raw_pcm_data, _ = process.communicate(input=clean_text.encode('utf-8'))

            if raw_pcm_data and len(raw_pcm_data) > 0:
                audio_array = np.frombuffer(raw_pcm_data, dtype=np.int16).astype(np.float32)
                length = len(audio_array)
                
                # --- SMOOTH, HIGH-FREQUENCY TAMED DSP PIPELINE ---
                t = np.arange(length)
                
                # 1. Softer throat chest resonance for the deep texture
                envelope = np.abs(audio_array) / 32768.0
                throat_resonance = np.sin(t * 0.09) * 90.0 * envelope
                
                # 2. Gentle, low-frequency vocal fry (smooth chest rasp without sharp noise)
                raspy_noise = np.random.normal(0, 110, length) * envelope * 0.25
                
                processed = audio_array + throat_resonance + raspy_noise
                
                # 3. Low-Pass Filter to completely kill sharp sounds, clicks, and harsh sibilance
                alpha_lp = 0.45  # Lower number = smoother, warmer, rounder sound
                smoothed = np.zeros_like(processed)
                smoothed[0] = processed[0]
                for i in range(1, length):
                    smoothed[i] = alpha_lp * processed[i] + (1 - alpha_lp) * smoothed[i-1]
                
                # 4. Clean volume normalization
                max_val = np.max(np.abs(smoothed))
                if max_val > 0:
                    smoothed = smoothed * (29000.0 / max_val)
                
                audio_array = np.clip(smoothed, -32768, 32767).astype(np.int16)
                
                # 5. Slightly higher playback rate (1.14x) to raise the pitch and smooth out cadence
                playback_rate = int(16000 * 1.14)
                # ---------------------------------------------

                total_samples = len(audio_array)
                total_words = len(words)
                audio_duration = max(0.1, total_samples / float(playback_rate))

                if worker_ref:
                    worker_ref.stream_start.emit()

                sd.play(audio_array, samplerate=playback_rate)
                
                start_time = time.time()
                emitted_words = 0
                while emitted_words < total_words:
                    elapsed = time.time() - start_time
                    target_index = int((elapsed / audio_duration) * total_words)
                    target_index = min(total_words, max(emitted_words, target_index))
                    
                    while emitted_words < target_index:
                        if worker_ref:
                            worker_ref.stream_word.emit(words[emitted_words] + " ")
                        emitted_words += 1
                    time.sleep(0.004)

                sd.wait()
                
                if worker_ref:
                    while emitted_words < total_words:
                        worker_ref.stream_word.emit(words[emitted_words] + " ")
                        emitted_words += 1
                    worker_ref.stream_end.emit()

                CURRENT_MIC_RMS = 0.0
                return
        except Exception as e:
            print(f"[Piper DSP Audio Error]: {e}")

    # Fallback TTS
    try:
        if worker_ref:
            worker_ref.stream_start.emit()
            for w in words:
                worker_ref.stream_word.emit(w + " ")
                time.sleep(0.035)
            worker_ref.stream_end.emit()

        import pyttsx3
        engine = pyttsx3.init('sapi5')
        engine.setProperty('rate', 190)
        engine.say(clean_text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        print(f"[TTS Fallback Error]: {e}")

print("Loading Whisper Speech Engine (base.en with VAD)... Please wait.")
try:
    stt_model = WhisperModel("base.en", device="cuda", compute_type="float16")
    print("[Whisper]: GPU acceleration enabled (CUDA).")
except Exception:
    stt_model = WhisperModel("base.en", device="cpu", compute_type="int8")
    print("[Whisper]: Running on CPU mode.")

WHISPER_PROMPT = (
    "E.V., Aimz, Sir, brightness, volume, mute, unmute, mute e.v, unmute e.v, battery, "
    "system specs, RAM, CPU, weather, search, look up, price, Bitcoin, Ethereum, "
    "Solana, dollar, rand, euro, pound, exchange rate, richest, net worth, YouTube, "
    "YouTube Music, play, pause, open, VS Code, Spotify, Discord, Explorer, Task Manager, "
    "ip address, network address, restart system, shut down system, search for file, "
    "find file, locate file, file search"
)

def listen(worker_ref=None, silence_limit=0.6, threshold=450):
    global CURRENT_MIC_RMS, IS_USER_TYPING, LAST_TYPING_TIME
    sample_rate = 16000
    chunk_size = 1024
    
    audio_chunks = []
    silent_chunks = 0
    max_silent_chunks = int(silence_limit * (sample_rate / chunk_size))
    has_spoken = False
    
    with sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16') as stream:
        while True:
            if worker_ref and worker_ref.is_muted:
                CURRENT_MIC_RMS = 0.0
                return ""

            if IS_USER_TYPING or (time.time() - LAST_TYPING_TIME < 1.2):
                sd.sleep(50)
                CURRENT_MIC_RMS = 0.0
                return ""

            data, _ = stream.read(chunk_size)
            audio_chunks.append(data.copy())
            
            rms = np.sqrt(np.mean(data.astype(np.float32)**2))
            CURRENT_MIC_RMS = float(rms)
            
            if rms > threshold:
                has_spoken = True
                silent_chunks = 0
            elif has_spoken:
                silent_chunks += 1
                if silent_chunks > max_silent_chunks:
                    break
            
            if len(audio_chunks) > int(4.5 * (sample_rate / chunk_size)) and not has_spoken:
                CURRENT_MIC_RMS = 0.0
                return ""

    CURRENT_MIC_RMS = 0.0
    if not has_spoken or not audio_chunks:
        return ""

    audio_data = np.concatenate(audio_chunks, axis=0)
    
    for _ in range(3):
        try:
            wavfile.write(temp_wav_path, sample_rate, audio_data)
            break
        except Exception:
            time.sleep(0.05)
            
    try:
        segments, _ = stt_model.transcribe(
            temp_wav_path, 
            language="en", 
            beam_size=1,
            best_of=1,
            vad_filter=True,
            repetition_penalty=1.2,
            condition_on_previous_text=False,
            initial_prompt=WHISPER_PROMPT,
            temperature=0.0
        )
        return "".join([segment.text for segment in segments]).strip()
    except Exception as e:
        print(f"[Whisper Transcribe Error]: {e}")
        return ""

# -------------------------------------------------------------
# 2. SYSTEM TOOLS & WORKSPACE UTILITIES
# -------------------------------------------------------------
APP_SHORTCUTS = {
    "vscode": "code",
    "vs code": "code",
    "visual studio code": "code",
    "spotify": "spotify",
    "discord": "discord",
    "notepad": "notepad",
    "calculator": "calc",
    "calc": "calc",
    "explorer": "explorer",
    "file explorer": "explorer",
    "files": "explorer",
    "task manager": "taskmgr",
    "terminal": "wt",
    "command prompt": "cmd",
    "cmd": "cmd",
    "settings": "ms-settings:"
}

def launch_application(app_name: str):
    clean_app = app_name.lower().strip()
    target_cmd = APP_SHORTCUTS.get(clean_app, clean_app)
    try:
        subprocess.Popen(f"start {target_cmd}", shell=True)
        return f"Opening {app_name}, Sir."
    except Exception as e:
        return f"Unable to launch {app_name}: {str(e)}"

def lock_workstation():
    ctypes.windll.user32.LockWorkStation()
    return "Workstation locked, Sir."

def minimize_all_windows():
    VK_LWIN = 0x5B
    VK_D = 0x44
    KEYEVENTF_KEYUP = 0x0002
    ctypes.windll.user32.keybd_event(VK_LWIN, 0, 0, 0)
    ctypes.windll.user32.keybd_event(VK_D, 0, 0, 0)
    ctypes.windll.user32.keybd_event(VK_D, 0, KEYEVENTF_KEYUP, 0)
    ctypes.windll.user32.keybd_event(VK_LWIN, 0, KEYEVENTF_KEYUP, 0)
    return "Showing desktop, Sir."

def get_battery_status():
    battery = psutil.sensors_battery()
    if battery is None:
        return "Battery telemetry is currently unavailable, Sir."
    percent = battery.percent
    plugged = "plugged into AC power" if battery.power_plugged else "running on battery"
    return f"Battery is at {percent}%, {plugged}, Sir."

def get_system_specs():
    ram = psutil.virtual_memory()
    cpu_usage = psutil.cpu_percent(interval=None)
    os_name = f"{platform.system()} {platform.release()}"
    total_ram_gb = round(ram.total / (1024 ** 3), 1)
    used_ram_gb = round(ram.used / (1024 ** 3), 1)
    return f"System running {os_name}. RAM usage is at {used_ram_gb} of {total_ram_gb} GB, with CPU load at {cpu_usage}%."

def open_browser():
    try:
        webbrowser.open("https://www.google.com")
        return "Opening your default browser now, Sir."
    except Exception as e:
        return f"Unable to launch browser: {str(e)}"

def play_youtube_music(query: str = ""):
    try:
        clean_q = query.strip()
        # Clean filler words
        clean_q = re.sub(r'^(play|search for|search|open)\s*(on)?\s*(youtube music|yt music)?\s*', '', clean_q)
        clean_q = re.sub(r'\s+(on|in)?\s*(youtube music|yt music|please)$', '', clean_q).strip()

        if not clean_q or clean_q in ["youtube music", "yt music", "music", "songs"]:
            webbrowser.open("https://music.youtube.com")
            return "Opening YouTube Music, Sir."

        if ytmusic_client:
            try:
                results = ytmusic_client.search(clean_q, filter="songs")
                if results and "videoId" in results[0]:
                    vid = results[0]["videoId"]
                    title = results[0].get("title", clean_q)
                    webbrowser.open(f"https://music.youtube.com/watch?v={vid}")
                    return f"Playing {title} on YouTube Music now, Sir."
            except Exception:
                pass

        encoded_query = urllib.parse.quote_plus(clean_q)
        webbrowser.open(f"https://music.youtube.com/search?q={encoded_query}")
        return f"Searching YouTube Music for {clean_q}, Sir."
    except Exception as e:
        return f"Unable to play track: {str(e)}"

def play_youtube_video(query: str = ""):
    try:
        clean_q = query.strip()
        
        # 1. Strip leading conversational request prefixes (e.g., "find me a", "show me", "can you look up")
        clean_q = re.sub(
            r'^(play|search for|search|look up|open|find|show me)\s+(me\s+)?(a|an)\s+(video|clip)?\s*(on\s+how\s+to|for|about)?\s*', 
            '', clean_q, flags=re.IGNORECASE
        )
        clean_q = re.sub(r'^(play|search for|search|look up|open|find|show me)\s*', '', clean_q, flags=re.IGNORECASE)
        
        # 2. Strip redundant platform or filler references at the start/end
        clean_q = re.sub(r'\s+(on|in)?\s*(youtube|yt|please|right now)$', '', clean_q, flags=re.IGNORECASE).strip()
        clean_q = re.sub(r'\byoutube\b', '', clean_q, flags=re.IGNORECASE).strip()
        
        # 3. Clean up dangling prepositions left over from phrasing like "on how to" or "video on"
        clean_q = re.sub(r'^(on how to|how to|on)\s+', '', clean_q, flags=re.IGNORECASE).strip()

        if not clean_q or clean_q.lower() in ["youtube", "yt", "videos"]:
            webbrowser.open("https://www.youtube.com")
            return "Opening YouTube, Sir."

        encoded_query = urllib.parse.quote_plus(clean_q)
        webbrowser.open(f"https://www.youtube.com/results?search_query={encoded_query}")
        return f"Searching YouTube for {clean_q}, Sir."
    except Exception as e:
        return f"Unable to launch YouTube video: {str(e)}"

def set_volume_level(level: int = 50):
    try:
        clamped_level = max(0, min(100, int(level)))
        devices = AudioUtilities.GetSpeakers()
        if hasattr(devices, 'EndpointVolume'):
            volume = devices.EndpointVolume
        else:
            from comtypes import CLSCTX_ALL, POINTER
            from pycaw.pycaw import IAudioEndpointVolume
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            volume = ctypes.cast(interface, POINTER(IAudioEndpointVolume))
            
        volume.SetMasterVolumeLevelScalar(clamped_level / 100.0, None)
        return f"Volume adjusted to {clamped_level}%, Sir."
    except Exception as e:
        return f"Volume adjustment failed: {str(e)}"

def set_brightness(level: int = 100):
    try:
        clamped_level = max(0, min(100, int(level)))
        try:
            sbc.set_brightness(clamped_level)
        except Exception:
            sbc.set_brightness(clamped_level, display=0)
        return f"Brightness adjusted to {clamped_level}%, Sir."
    except Exception as e:
        return f"Brightness control unavailable or failed: {str(e)}"

def media_control(action: str = "play_pause"):
    VK_MEDIA_NEXT_TRACK = 0xB0
    VK_MEDIA_PREV_TRACK = 0xB1
    VK_MEDIA_PLAY_PAUSE = 0xB3
    KEYEVENTF_EXTENDEDKEY = 0x0001
    KEYEVENTF_KEYUP = 0x0002

    key_map = {
        "play_pause": VK_MEDIA_PLAY_PAUSE,
        "play": VK_MEDIA_PLAY_PAUSE,
        "pause": VK_MEDIA_PLAY_PAUSE,
        "next": VK_MEDIA_NEXT_TRACK,
        "previous": VK_MEDIA_PREV_TRACK,
        "prev": VK_MEDIA_PREV_TRACK
    }

    key = key_map.get(action.lower().strip(), VK_MEDIA_PLAY_PAUSE)
    ctypes.windll.user32.keybd_event(key, 0, KEYEVENTF_EXTENDEDKEY, 0)
    ctypes.windll.user32.keybd_event(key, 0, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0)
    return "Media command executed, Sir."

def get_live_weather(location=""):
    try:
        query_loc = location.strip() if location else ""
        url = f"https://wttr.in/{query_loc}?format=%C:+%t+(feels+like+%f)"
        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            return f"Weather report: {response.text.strip()}, Sir."
    except Exception:
        pass
    return None

def get_crypto_price(coin="bitcoin"):
    try:
        coin_clean = coin.lower().strip()
        alias_map = {
            "btc": "bitcoin",
            "bitcoin": "bitcoin",
            "eth": "ethereum",
            "ethereum": "ethereum",
            "sol": "solana",
            "solana": "solana",
            "xrp": "ripple",
            "doge": "dogecoin",
            "dogecoin": "dogecoin",
            "ada": "cardano",
            "cardano": "cardano"
        }
        coin_id = alias_map.get(coin_clean, coin_clean)
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={urllib.parse.quote(coin_id)}&vs_currencies=usd"
        res = requests.get(url, timeout=3).json()
        if coin_id in res and "usd" in res[coin_id]:
            price = res[coin_id]["usd"]
            return f"{coin_id.capitalize()} is currently trading at ${price:,.2f} USD, Sir."
    except Exception:
        pass
    return None

def get_exchange_rate(base="USD", target="ZAR"):
    try:
        base_clean = base.upper().strip()
        target_clean = target.upper().strip()
        url = f"https://api.frankfurter.app/latest?from={base_clean}&to={target_clean}"
        res = requests.get(url, timeout=3).json()
        if "rates" in res and target_clean in res["rates"]:
            rate = res["rates"][target_clean]
            return f"The current exchange rate for 1 {base_clean} is {rate:,.2f} {target_clean}, Sir."
    except Exception:
        pass
    return None

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return f"Local network IP address is {ip}, Sir."
    except Exception:
        return "Unable to retrieve local IP telemetry, Sir."

def search_local_files(query: str, search_root: str = None) -> list:
    """Searches local directories for files matching the query string."""
    if not search_root:
        user_profile = os.path.expanduser("~")
        search_roots = [
            os.path.join(user_profile, "Desktop"),
            os.path.join(user_profile, "Documents"),
            os.path.join(user_profile, "Downloads")
        ]
    else:
        search_roots = [search_root]

    matches = []
    clean_query = query.lower().strip()

    for root_dir in search_roots:
        if not os.path.exists(root_dir):
            continue
        try:
            for dirpath, _, filenames in os.walk(root_dir):
                for filename in filenames:
                    if clean_query in filename.lower():
                        full_path = os.path.join(dirpath, filename)
                        matches.append(full_path)
                        if len(matches) >= 5:
                            break
                if len(matches) >= 5:
                    break
        except Exception as e:
            print(f"[File Search Error in {root_dir}]: {e}")

    return matches

def handle_file_search_command(clean_input: str):
    """Parses search/open requests for local files with flexible phrasing."""
    q = re.sub(r'^(search for file|find file|locate file|file search|locate|find)\s*', '', clean_input).strip()
    q = re.sub(r'\s+file$', '', q).strip()
    
    if not q:
        return "Please specify a file name to search for, Sir."

    matches = search_local_files(q)
    if not matches:
        return f"No local files matching '{q}' were found, Sir."

    top_match = matches[0]
    file_name = os.path.basename(top_match)
    
    try:
        os.startfile(top_match)
        return f"Located and opened {file_name}, Sir."
    except Exception as e:
        return f"Found {file_name}, but failed to launch it: {str(e)}"

def web_search(query: str):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if results:
                snippets = [f"{r['title']}: {r['body']}" for r in results]
                return "\n".join(snippets)
    except Exception as e:
        return f"Web search encountered an issue: {str(e)}"
    return "No relevant web search results found."

def handle_direct_commands(text):
    clean = text.lower()
    
    # Strict file search triggers to prevent false positives
    file_triggers = ["search for file", "find file", "locate file", "file search"]
    if any(clean.startswith(prefix) for prefix in file_triggers):
        return handle_file_search_command(clean)

    if "volume" in clean or "sound" in clean:
        nums = re.findall(r'\b\d+\b', clean)
        if nums:
            return set_volume_level(int(nums[0]))
        if "mute" in clean:
            return set_volume_level(0)
        if "max" in clean or "full" in clean:
            return set_volume_level(100)

    if "brightness" in clean or "dim" in clean:
        nums = re.findall(r'\b\d+\b', clean)
        if nums:
            return set_brightness(int(nums[0]))
        if "max" in clean or "full" in clean:
            return set_brightness(100)
        if "dim" in clean:
            return set_brightness(30)

    if any(k in clean for k in ["lock pc", "lock workstation", "lock computer", "lock screen"]):
        return lock_workstation()
        
    if any(k in clean for k in ["show desktop", "minimize all", "minimize windows", "clear screen"]):
        return minimize_all_windows()

    if any(k in clean for k in ["pause", "resume", "unpause"]):
        return media_control("play_pause")
    if any(w in clean for w in ["next track", "next song", "skip track", "skip song"]):
        return media_control("next")
    if any(w in clean for w in ["previous track", "previous song", "back song"]):
        return media_control("previous")

    if any(k in clean for k in ["youtube music", "yt music"]) or (clean.startswith("play") and any(w in clean for w in ["song", "track", "music"])):
        return play_youtube_music(text)

    if "youtube" in clean or "yt" in clean or clean.startswith("play video") or clean.startswith("search youtube"):
        return play_youtube_video(text)

    if "battery" in clean or "power" in clean or "charge" in clean:
        return get_battery_status()
        
    if "spec" in clean or "ram" in clean or "cpu" in clean or "system" in clean:
        return get_system_specs()
        
    if "browser" in clean or "chrome" in clean or "google" in clean or "open internet" in clean:
        return open_browser()

    if any(clean.startswith(prefix) for prefix in ["open ", "launch ", "start "]):
        app_target = re.sub(r'^(open|launch|start)\s+(app|application)?\s*', '', clean).strip()
        if app_target:
            return launch_application(app_target)

    if "bitcoin" in clean or "btc" in clean:
        return get_crypto_price("bitcoin")
    if "ethereum" in clean or "eth" in clean:
        return get_crypto_price("ethereum")
    if "solana" in clean or "sol" in clean:
        return get_crypto_price("solana")
    if "ripple" in clean or "xrp" in clean:
        return get_crypto_price("ripple")
    if "doge" in clean:
        return get_crypto_price("dogecoin")
    if "cardano" in clean or "ada" in clean:
        return get_crypto_price("cardano")

    if any(k in clean for k in ["dollar to rand", "usd to zar", "dollar rand"]):
        return get_exchange_rate("USD", "ZAR")
    if any(k in clean for k in ["euro to dollar", "eur to usd"]):
        return get_exchange_rate("EUR", "USD")
    if any(k in clean for k in ["pound to dollar", "gbp to usd"]):
        return get_exchange_rate("GBP", "USD")
    if any(k in clean for k in ["pound to rand", "gbp to zar"]):
        return get_exchange_rate("GBP", "ZAR")
    if any(k in clean for k in ["euro to rand", "eur to zar"]):
        return get_exchange_rate("EUR", "ZAR")

    if "ip address" in clean or "network address" in clean:
        return get_local_ip()

    if "restart system" in clean or "reboot computer" in clean:
        os.system("shutdown /r /t 5")
        return "Initiating system reboot sequence, Sir."

    if "shut down system" in clean or "power off computer" in clean:
        os.system("shutdown /s /t 5")
        return "Initiating system shutdown sequence, Sir."

    if "weather" in clean:
        match = re.search(r'weather\s+(?:in|for)?\s*([a-zA-Z\s]+)', clean)
        loc = match.group(1).strip() if match else ""
        weather_res = get_live_weather(loc)
        if weather_res:
            return weather_res

    return None

# -------------------------------------------------------------
# 3. BACKGROUND ASSISTANT THREAD
# -------------------------------------------------------------
class AssistantWorker(QThread):
    status_changed = pyqtSignal(str)
    user_speech_detected = pyqtSignal(str)
    app_shutdown_requested = pyqtSignal()
    
    stream_start = pyqtSignal()
    stream_word = pyqtSignal(str)
    stream_end = pyqtSignal()
    
    toggle_mute_requested = pyqtSignal()
    busy_state_changed = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.is_running = True
        self.is_muted = False
        self.voice_silent_mode = False
        self.is_busy = False
        self.text_queue = deque()

    def submit_text_query(self, text):
        self.text_queue.append(text)

    def run(self):
        self.status_changed.emit("SPEAKING")
        init_greeting = "Cybernetic interface initialized. Systems nominal, Sir."
        speak(init_greeting, self, force_speech=True)
        
        if self.is_muted:
            self.status_changed.emit("MUTED")
        else:
            self.status_changed.emit("STANDBY")

        system_prompt = {
            "role": "system", 
            "content": (
                "You are J.A.R.V.I.S., operating under the designation E.V. "
                "You are an elite, highly articulate digital assistant. "
                "Address the user as Sir approximately 65% of the time, and as Aimz the remaining 35% of the time, choosing dynamically. "
                "Maintain a dry, sophisticated wit, absolute technical competence, and a brisk, professional cadence. "
                "Never break character. Keep answers strictly to 1 concise sentence. "
                "Use the provided live web search context directly to answer."
            )
        }
        messages = [system_prompt]
        SEARCH_TRIGGERS = [
            "search", "google", "look up", "who is", "who's", "what is", "what's",
            "richest", "wealthiest", "latest", "news", "score", "when is", "where is",
            "tell me about", "current", "currently", "how much is", "net worth", "today"
        ]

        while self.is_running:
            try:
                if len(self.text_queue) > 0:
                    user_input = self.text_queue.popleft()
                elif self.is_muted:
                    self.status_changed.emit("MUTED")
                    time.sleep(0.2)
                    continue
                else:
                    self.status_changed.emit("LISTENING")
                    user_input = listen(worker_ref=self, silence_limit=0.6, threshold=450)

                if not user_input or len(user_input.strip()) < 2:
                    if self.is_muted:
                        self.status_changed.emit("MUTED")
                    else:
                        self.status_changed.emit("STANDBY")
                    continue

                self.is_busy = True
                self.busy_state_changed.emit(True)
                self.user_speech_detected.emit(user_input)
                clean_input = user_input.lower().replace(".", "").strip()

                if "mute e.v" in clean_input or "mute ev" in clean_input:
                    self.toggle_mute_requested.emit()
                    self.is_busy = False
                    self.busy_state_changed.emit(False)
                    continue

                if any(w in clean_input for w in ["shutdown", "shut down", "goodbye", "exit", "stop"]):
                    self.status_changed.emit("SPEAKING")
                    speak("Systems going offline. Goodbye, Sir.", self, force_speech=True)
                    self.is_running = False
                    self.app_shutdown_requested.emit()
                    break

                self.status_changed.emit("PROCESSING")
                fast_reply = handle_direct_commands(user_input)
                
                if fast_reply:
                    self.status_changed.emit("SPEAKING")
                    messages.append({"role": "user", "content": user_input})
                    messages.append({"role": "assistant", "content": fast_reply})
                    speak(fast_reply, self)
                    
                    if self.is_muted:
                        self.status_changed.emit("MUTED")
                    else:
                        self.status_changed.emit("STANDBY")
                        
                    self.is_busy = False
                    self.busy_state_changed.emit(False)
                    continue

                needs_search = any(trigger in clean_input for trigger in SEARCH_TRIGGERS)
                search_context = ""
                if needs_search:
                    query = re.sub(r'^(search for|search|look up|what is|what\'s|whats|who is|who\'s|whos|tell me about)\s+', '', clean_input).strip()
                    search_results = web_search(query)
                    search_context = f"\n\nLIVE SEARCH RESULTS:\n{search_results}"

                user_message_content = user_input + search_context
                messages.append({"role": "user", "content": user_message_content})
                if len(messages) > 5:
                    messages = [system_prompt] + messages[-3:]

                response = ollama.chat(model='llama3.2:3b', messages=messages)
                reply = response.message.content

                self.status_changed.emit("SPEAKING")
                messages.append({"role": "assistant", "content": reply})
                speak(reply, self)
                
                if self.is_muted:
                    self.status_changed.emit("MUTED")
                else:
                    self.status_changed.emit("STANDBY")
                    
                self.is_busy = False
                self.busy_state_changed.emit(False)

            except Exception as e:
                print(f"[Loop Exception]: {e}")
                if self.is_muted:
                    self.status_changed.emit("MUTED")
                else:
                    self.status_changed.emit("STANDBY")
                self.is_busy = False
                self.busy_state_changed.emit(False)


# -------------------------------------------------------------
# 4. HIGH-DENSITY NEURAL SYNAPSE SWARM (DENSE MULTI-CLUSTER)
# -------------------------------------------------------------
class NeuralSynapseSwarm(QWidget):
    clicked_mute = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(310, 310)
        self.setMouseTracking(True)
        self.state = "STANDBY"

        self.cluster_centers = [
            {'x': -0.75, 'y':  0.35, 'z':  0.25},
            {'x':  0.75, 'y':  0.35, 'z': -0.25},
            {'x': -0.45, 'y': -0.65, 'z': -0.35},
            {'x':  0.45, 'y': -0.65, 'z':  0.35},
            {'x':  0.00, 'y':  0.75, 'z':  0.50},
            {'x':  0.00, 'y': -0.15, 'z': -0.65},
        ]

        self.num_nodes = 260
        self.nodes = []
        for i in range(self.num_nodes):
            cluster = self.cluster_centers[i % len(self.cluster_centers)]
            if random.random() < 0.72:
                spread = random.gauss(0, 0.32)
                x = cluster['x'] + spread
                y = cluster['y'] + spread
                z = cluster['z'] + spread
            else:
                u = random.uniform(0, 1)
                theta = random.uniform(0, 2 * math.pi)
                phi = math.acos(2 * u - 1)
                r = random.uniform(0.35, 1.45)
                x = r * math.sin(phi) * math.cos(theta)
                y = r * math.sin(phi) * math.sin(theta)
                z = r * math.cos(phi)

            self.nodes.append({
                'base_x': x, 'base_y': y, 'base_z': z,
                'orbit_speed': random.uniform(0.5, 1.4),
                'phase': random.uniform(0, 2 * math.pi),
                'is_super': (i % 14 == 0),
                'pulse_offset': random.uniform(0, math.pi * 2)
            })

        self.rot_y = 0.0
        self.rot_x = 0.0
        self.target_tilt_x = 0.0
        self.target_tilt_y = 0.0
        self.curr_tilt_x = 0.0
        self.curr_tilt_y = 0.0

        self.swarm_expansion = 1.0
        self.vortex_collapse = 1.0
        self.shockwave_phase = 0.0
        self.ring_rot = [0.0, 90.0]

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_physics)
        self.timer.start(20)

    def set_state(self, state):
        self.state = state

    def set_parallax(self, norm_x, norm_y):
        self.target_tilt_y = norm_x * 0.40
        self.target_tilt_x = -norm_y * 0.40

    def update_physics(self):
        global CURRENT_MIC_RMS
        
        if self.state == "PROCESSING":
            speed_mult = 3.8
            target_collapse = 0.48
            target_exp = 1.0
        elif self.state == "LISTENING":
            speed_mult = 1.2
            target_collapse = 1.0
            energy = min(1.0, CURRENT_MIC_RMS / 2000.0)
            target_exp = 1.0 + (energy * 0.65)
        elif self.state == "SPEAKING":
            speed_mult = 2.0
            target_collapse = 1.0
            target_exp = 1.0 + math.sin(self.shockwave_phase) * 0.18
            self.shockwave_phase += 0.28
        else:
            speed_mult = 0.8
            target_collapse = 1.0
            target_exp = 1.0 + math.sin(time.time() * 1.5) * 0.05

        self.rot_y = (self.rot_y + 0.013 * speed_mult) % (2 * math.pi)
        self.rot_x = (self.rot_x + 0.007 * speed_mult) % (2 * math.pi)

        self.curr_tilt_x += (self.target_tilt_x - self.curr_tilt_x) * 0.1
        self.curr_tilt_y += (self.target_tilt_y - self.curr_tilt_y) * 0.1

        self.swarm_expansion += (target_exp - self.swarm_expansion) * 0.25
        self.vortex_collapse += (target_collapse - self.vortex_collapse) * 0.2

        self.ring_rot[0] = (self.ring_rot[0] + 0.9 * speed_mult) % 360
        self.ring_rot[1] = (self.ring_rot[1] - 1.3 * speed_mult) % 360

        self.update()

    def mousePressEvent(self, event):
        cx, cy = self.width() / 2, self.height() / 2
        if math.hypot(event.position().x() - cx, event.position().y() - cy) < 60:
            self.clicked_mute.emit()
            event.accept()

    def project_node(self, x, y, z, cx, cy, radius_scale, rx, ry):
        x1 = x * math.cos(ry) + z * math.sin(ry)
        y1 = y
        z1 = -x * math.sin(ry) + z * math.cos(ry)

        x2 = x1
        y2 = y1 * math.cos(rx) - z1 * math.sin(rx)
        z2 = y1 * math.sin(rx) + z1 * math.cos(rx)

        fov = 240.0
        depth = fov / (fov + z2 * radius_scale)
        px = cx + x2 * radius_scale * depth
        py = cy + y2 * radius_scale * depth
        return px, py, z2

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        cx, cy = self.width() / 2, self.height() / 2

        colors = {
            "STANDBY": (QColor(0, 210, 255), QColor(0, 210, 255, 18)),
            "LISTENING": (QColor(0, 255, 170), QColor(0, 255, 170, 26)),
            "PROCESSING": (QColor(255, 190, 0), QColor(255, 190, 0, 26)),
            "SPEAKING": (QColor(0, 160, 255), QColor(0, 160, 255, 30)),
            "MUTED": (QColor(130, 140, 150), QColor(130, 140, 150, 12))
        }
        primary, glow = colors.get(self.state, colors["STANDBY"])

        painter.save()
        painter.translate(cx + (self.curr_tilt_y * 12), cy + (self.curr_tilt_x * 12))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        
        pen_ring1 = QPen(QColor(primary.red(), primary.green(), primary.blue(), 45), 1, Qt.PenStyle.DashLine)
        painter.setPen(pen_ring1)
        painter.rotate(self.ring_rot[0])
        painter.drawEllipse(-122, -44, 244, 88)

        pen_ring2 = QPen(QColor(primary.red(), primary.green(), primary.blue(), 35), 1, Qt.PenStyle.DotLine)
        painter.setPen(pen_ring2)
        painter.rotate(self.ring_rot[1])
        painter.drawEllipse(-108, -62, 216, 124)
        painter.restore()

        tot_rx = self.rot_x + self.curr_tilt_x
        tot_ry = self.rot_y + self.curr_tilt_y
        effective_r = 82.0 * self.swarm_expansion * self.vortex_collapse

        proj_nodes = []
        for nd in self.nodes:
            wobble = math.sin(time.time() * nd['orbit_speed'] + nd['phase']) * 0.06
            nx = nd['base_x'] + wobble
            ny = nd['base_y'] + wobble
            nz = nd['base_z'] + wobble

            px, py, pz = self.project_node(nx, ny, nz, cx, cy, effective_r, tot_rx, tot_ry)
            proj_nodes.append((px, py, pz, nd['is_super'], nd['pulse_offset']))

        if self.state != "MUTED":
            conn_dist_sq = (46.0 * self.vortex_collapse) ** 2
            for i in range(0, len(proj_nodes)):
                p1 = proj_nodes[i]
                for j in range(i + 1, min(i + 8, len(proj_nodes))):
                    p2 = proj_nodes[j]
                    dx = p1[0] - p2[0]
                    dy = p1[1] - p2[1]
                    d_sq = dx*dx + dy*dy
                    if d_sq < conn_dist_sq:
                        dist = math.sqrt(d_sq)
                        max_d = 46.0 * self.vortex_collapse
                        alpha = int(max(10, min(160, (1.0 - (dist / max_d)) * 130)))
                        avg_z = (p1[2] + p2[2]) / 2.0
                        if avg_z < 0: alpha = int(alpha * 0.45)

                        pen_line = QPen(QColor(primary.red(), primary.green(), primary.blue(), alpha), 0.8)
                        painter.setPen(pen_line)
                        painter.drawLine(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]))

        for (px, py, pz, is_super, p_offset) in proj_nodes:
            depth_alpha = int(max(25, min(240, 130 + pz * 110)))
            
            if is_super and self.state != "MUTED":
                pulse_brightness = (math.sin(time.time() * 6.5 + p_offset) + 1.0) * 0.5
                painter.setBrush(QBrush(QColor(255, 235, 170, int(180 + pulse_brightness * 75))))
                painter.setPen(QPen(primary, 1.2))
                sz = 3.6 if pz > 0 else 2.2
                painter.drawEllipse(int(px - sz/2), int(py - sz/2), int(sz), int(sz))
            else:
                painter.setBrush(QBrush(QColor(primary.red(), primary.green(), primary.blue(), depth_alpha)))
                painter.setPen(Qt.PenStyle.NoPen)
                sz = 2.0 if pz > 0 else 1.2
                painter.drawEllipse(int(px - sz/2), int(py - sz/2), int(sz), int(sz))

        core_grad = QRadialGradient(cx, cy, 26)
        core_grad.setColorAt(0.0, QColor(255, 255, 255, 240))
        core_grad.setColorAt(0.45, primary)
        core_grad.setColorAt(1.0, QColor(6, 12, 20, 230))
        painter.setBrush(QBrush(core_grad))
        painter.setPen(QPen(primary, 1.2))
        painter.drawEllipse(int(cx - 24), int(cy - 24), 48, 48)

        painter.setPen(QPen(QColor(0, 0, 0, 230) if self.state != "MUTED" else QColor(255, 255, 255, 220)))
        painter.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        core_lbl = "MUTE" if self.state == "MUTED" else "E.V."
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, core_lbl)


# -------------------------------------------------------------
# 5. CHAMFERED OSCILLOSCOPE TELEMETRY GRAPH
# -------------------------------------------------------------
class CyberSparklineGraph(QWidget):
    def __init__(self, title, max_history=36, parent=None):
        super().__init__(parent)
        self.setFixedSize(148, 92)
        self.title = title
        self.history = deque([0.0] * max_history, maxlen=max_history)
        self.subtext = "0%"
        self.primary_color = QColor(0, 210, 255)

    def add_data_point(self, val, subtext=""):
        self.history.append(min(100.0, max(0.0, float(val))))
        self.subtext = subtext
        self.update()

    def set_theme_color(self, color):
        self.primary_color = color
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        chamfer_path = QPainterPath()
        chamfer_path.moveTo(6, 0)
        chamfer_path.lineTo(w - 6, 0)
        chamfer_path.lineTo(w, 6)
        chamfer_path.lineTo(w, h - 6)
        chamfer_path.lineTo(w - 6, h)
        chamfer_path.lineTo(6, h)
        chamfer_path.lineTo(0, h - 6)
        chamfer_path.lineTo(0, 6)
        chamfer_path.closeSubpath()

        painter.fillPath(chamfer_path, QBrush(QColor(4, 10, 18, 140)))
        painter.setPen(QPen(QColor(self.primary_color.red(), self.primary_color.green(), self.primary_color.blue(), 90), 1))
        painter.drawPath(chamfer_path)

        painter.setPen(QPen(QColor(240, 245, 255, 235)))
        painter.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        painter.drawText(6, 12, f"[{self.title}]")

        painter.setPen(QPen(self.primary_color))
        painter.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
        painter.drawText(w - 48, 12, f"~ {self.subtext}")

        gx = 24
        gy = 18
        gw = w - gx - 6
        gh = h - gy - 8

        painter.fillRect(gx, gy, gw, gh, QColor(4, 10, 18, 110))
        painter.setPen(QPen(QColor(self.primary_color.red(), self.primary_color.green(), self.primary_color.blue(), 100), 1))
        painter.drawRect(gx, gy, gw, gh)

        painter.setPen(QPen(QColor(160, 185, 205, 180)))
        painter.setFont(QFont("Consolas", 5))
        painter.drawText(2, gy + 7, "100")
        painter.drawText(6, int(gy + gh / 2 + 3), "50")
        painter.drawText(10, gy + gh, "0")

        grid_pen = QPen(QColor(255, 255, 255, 24), 1, Qt.PenStyle.DotLine)
        painter.setPen(grid_pen)
        for i in range(1, 4):
            y_line = int(gy + (gh * i / 4.0))
            painter.drawLine(gx, y_line, gx + gw, y_line)

        for i in range(1, 6):
            x_line = int(gx + (gw * i / 6.0))
            painter.drawLine(x_line, gy, x_line, gy + gh)

        pts = []
        n = len(self.history)
        for i, val in enumerate(self.history):
            px = gx + (float(i) / (n - 1)) * gw
            py = (gy + gh) - (val / 100.0) * gh
            pts.append(QPointF(px, py))

        if len(pts) > 1:
            fill_path = QPainterPath()
            fill_path.moveTo(pts[0].x(), gy + gh)
            for pt in pts:
                fill_path.lineTo(pt)
            fill_path.lineTo(pts[-1].x(), gy + gh)
            fill_path.closeSubpath()

            grad = QLinearGradient(0, gy, 0, gy + gh)
            grad.setColorAt(0.0, QColor(self.primary_color.red(), self.primary_color.green(), self.primary_color.blue(), 85))
            grad.setColorAt(0.7, QColor(self.primary_color.red(), self.primary_color.green(), self.primary_color.blue(), 20))
            grad.setColorAt(1.0, QColor(0, 0, 0, 0))
            painter.fillPath(fill_path, QBrush(grad))

            wave_pen = QPen(self.primary_color, 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
            painter.setPen(wave_pen)
            for i in range(len(pts) - 1):
                painter.drawLine(pts[i], pts[i+1])

            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.setPen(QPen(self.primary_color, 1))
            painter.drawEllipse(pts[-1], 2.2, 2.2)


# -------------------------------------------------------------
# 6. CHAMFERED TACTICAL PUSH BUTTON WIDGET
# -------------------------------------------------------------
class CyberChamferButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setFixedHeight(28)
        self.primary_color = QColor(0, 210, 255)
        self.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_theme_color(self, color):
        self.primary_color = color
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        is_down = self.isDown()
        is_hover = self.underMouse()

        path = QPainterPath()
        path.moveTo(6, 0)
        path.lineTo(w - 6, 0)
        path.lineTo(w, 6)
        path.lineTo(w, h - 6)
        path.lineTo(w - 6, h)
        path.lineTo(6, h)
        path.lineTo(0, h - 6)
        path.lineTo(0, 6)
        path.closeSubpath()

        bg_alpha = 75 if is_down else (50 if is_hover else 25)
        painter.fillPath(path, QBrush(QColor(self.primary_color.red(), self.primary_color.green(), self.primary_color.blue(), bg_alpha)))

        border_pen = QPen(self.primary_color if is_hover else QColor(self.primary_color.red(), self.primary_color.green(), self.primary_color.blue(), 120), 1.2)
        painter.setPen(border_pen)
        painter.drawPath(path)

        painter.setPen(QPen(QColor(255, 255, 255) if is_hover else QColor(220, 235, 250)))
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.text())


# -------------------------------------------------------------
# 7. COMMAND LINE PROMPT (KEYBOARD HISTORY NAVIGATION)
# -------------------------------------------------------------
class CyberCommandLine(QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.history = []
        self.history_index = 0

    def add_to_history(self, text):
        if text and (not self.history or self.history[-1] != text):
            self.history.append(text)
        self.history_index = len(self.history)

    def keyPressEvent(self, event: QKeyEvent):
        global IS_USER_TYPING, LAST_TYPING_TIME
        IS_USER_TYPING = True
        LAST_TYPING_TIME = time.time()

        if event.key() == Qt.Key.Key_Up:
            if self.history and self.history_index > 0:
                self.history_index -= 1
                self.setText(self.history[self.history_index])
            event.accept()
            return
        elif event.key() == Qt.Key.Key_Down:
            if self.history and self.history_index < len(self.history) - 1:
                self.history_index += 1
                self.setText(self.history[self.history_index])
            else:
                self.history_index = len(self.history)
                self.clear()
            event.accept()
            return
        elif event.key() == Qt.Key.Key_Escape:
            self.clear()
            event.accept()
            return

        super().keyPressEvent(event)


# -------------------------------------------------------------
# 8. LIVE HARDWARE [KERN_STREAM // MEM_DUMP] PANEL
# -------------------------------------------------------------
class CyberMemoryStreamPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.state = "STANDBY"
        self.primary_color = QColor(0, 210, 255)
        
        self.lines = deque(maxlen=20)
        self.init_hardware_stream()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.stream_step)
        self.timer.start(120)

    def init_hardware_stream(self):
        for _ in range(20):
            self.lines.append(self.sample_hardware_memory_dump())

    def set_state(self, state):
        self.state = state
        if state == "PROCESSING":
            self.timer.setInterval(30)
        elif state == "SPEAKING":
            self.timer.setInterval(60)
        else:
            self.timer.setInterval(120)

    def set_theme_color(self, color):
        self.primary_color = color
        self.update()

    def sample_hardware_memory_dump(self):
        try:
            pids = psutil.pids()
            if pids:
                pid = random.choice(pids)
                proc = psutil.Process(pid)
                mem_info = proc.memory_info()
                rss_bytes = mem_info.rss
                vms_bytes = mem_info.vms
                
                addr = f"0x{(pid * 0x10000 + (vms_bytes & 0xFFFF)):08X}"
                
                b1 = (rss_bytes >> 24) & 0xFF
                b2 = (rss_bytes >> 16) & 0xFF
                b3 = (rss_bytes >> 8) & 0xFF
                b4 = rss_bytes & 0xFF
                b5 = (vms_bytes >> 16) & 0xFF
                b6 = (vms_bytes >> 8) & 0xFF
                hex_bytes = f"{b1:02X} {b2:02X} {b3:02X} {b4:02X} {b5:02X} {b6:02X}"
                
                pname = proc.name()[:8].ljust(8, '.')
                ascii_chars = f"[{proc.status()[:3].upper()}] {pname}"
                return addr, hex_bytes, ascii_chars
        except Exception:
            pass

        addr = f"0x{random.randint(0x10000000, 0x7FFFFFFF):08X}"
        hex_bytes = " ".join([f"{random.randint(0, 255):02X}" for _ in range(6)])
        ascii_chars = "[ACT] sys_idle"
        return addr, hex_bytes, ascii_chars

    def stream_step(self):
        self.lines.append(self.sample_hardware_memory_dump())
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        path = QPainterPath()
        path.moveTo(0, 0)
        path.lineTo(w - 48, 0)
        path.lineTo(w - 32, 16)
        path.lineTo(w, 16)
        path.lineTo(w, h)
        path.lineTo(0, h)
        path.lineTo(0, 72)
        path.lineTo(12, 60)
        path.lineTo(12, 28)
        path.lineTo(0, 16)
        path.closeSubpath()

        painter.fillPath(path, QBrush(QColor(4, 10, 18, 140)))
        border_pen = QPen(QColor(self.primary_color.red(), self.primary_color.green(), self.primary_color.blue(), 110), 1.2)
        painter.setPen(border_pen)
        painter.drawPath(path)

        grill_pen = QPen(QColor(self.primary_color.red(), self.primary_color.green(), self.primary_color.blue(), 140), 1)
        painter.setPen(grill_pen)
        for i in range(7):
            gx = (w - 44) + i * 5
            painter.drawLine(gx, 4, gx + 8, 12)

        painter.setPen(QPen(QColor(235, 245, 255, 240)))
        painter.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        painter.drawText(16, 18, "[KERN_STREAM // MEM_DUMP]")

        painter.setBrush(QBrush(QColor(self.primary_color.red(), self.primary_color.green(), self.primary_color.blue(), 160)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(w - 14, 26, 4, 18, 2, 2)

        painter.setFont(QFont("Consolas", 7))
        y = 38
        for idx, (addr, hex_b, ascii_c) in enumerate(self.lines):
            alpha = int(70 + (idx / len(self.lines)) * 185)
            
            painter.setPen(QPen(QColor(130, 175, 215, alpha)))
            painter.drawText(14, y, f"{addr}")

            painter.setPen(QPen(QColor(self.primary_color.red(), self.primary_color.green(), self.primary_color.blue(), alpha)))
            painter.drawText(86, y, hex_b)

            painter.setPen(QPen(QColor(160, 205, 225, alpha)))
            painter.drawText(w - 92, y, ascii_c)

            y += 13


# -------------------------------------------------------------
# 9. MAIN WIDESCREEN COMMAND HUD WINDOW
# -------------------------------------------------------------
class JarvisWidescreenHUD(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("E.V. Cybernetic Tactical Interface")
        self.setFixedSize(1060, 610)
        self.setMouseTracking(True)
        
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self.drag_position = QPoint()
        self.is_dragging = False
        self.is_compact = False
        self.current_theme = QColor(0, 210, 255)

        self.init_ui()

        self.worker = AssistantWorker()
        self.worker.status_changed.connect(self.update_status)
        self.worker.user_speech_detected.connect(self.append_user_transcript)
        
        self.worker.stream_start.connect(self.handle_stream_start)
        self.worker.stream_word.connect(self.handle_stream_word)
        self.worker.stream_end.connect(self.handle_stream_end)

        self.worker.toggle_mute_requested.connect(self.toggle_mute)
        self.worker.busy_state_changed.connect(self.update_busy_ui)
        self.worker.app_shutdown_requested.connect(self.close_application)
        self.worker.start()

        self.typing_timer = QTimer(self)
        self.typing_timer.timeout.connect(self.check_typing_status)
        self.typing_timer.start(300)

        self.stats_timer = QTimer(self)
        self.stats_timer.timeout.connect(self.update_telemetry)
        self.stats_timer.start(1000)
        self.update_telemetry()

    def init_ui(self):
        self.main_container = QWidget(self)
        self.main_container.setObjectName("Container")
        self.main_container.setMouseTracking(True)
        
        self.main_container.setStyleSheet("""
            QWidget#Container {
                background-color: rgba(6, 12, 20, 0.94);
                border: 1.5px solid rgba(0, 210, 255, 0.5);
                border-radius: 18px;
            }
            QFrame {
                border: none !important;
                background: transparent !important;
            }
            QLabel {
                font-family: 'Consolas', monospace;
                border: none !important;
                background: transparent !important;
            }
            QTextEdit {
                background-color: transparent !important;
                border: none !important;
                color: #00f0ff;
                font-family: 'Consolas', monospace;
                font-size: 11px;
                padding: 4px;
            }
            QLineEdit {
                background-color: rgba(4, 10, 18, 140);
                border: 1px solid rgba(0, 210, 255, 0.4);
                border-radius: 4px;
                color: #00f0ff;
                font-family: 'Consolas', monospace;
                font-size: 10px;
                padding: 4px 8px;
            }
            QLineEdit:focus {
                border: 1px solid #00d2ff;
                background-color: rgba(4, 10, 18, 200);
            }
        """)

        root_layout = QVBoxLayout(self.main_container)
        root_layout.setContentsMargins(18, 12, 18, 12)
        root_layout.setSpacing(6)

        top_bar = QHBoxLayout()
        self.title_label = QLabel("MARK-VIII // E.V. TACTICAL CYBERDECK")
        self.title_label.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
        self.title_label.setStyleSheet("color: #00d2ff; letter-spacing: 2px;")
        
        self.badge_status = QLabel("[SYS_OK]")
        self.badge_status.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        self.badge_status.setStyleSheet("color: #00ffaa; background: rgba(0,255,170,0.12); padding: 2px 6px; border-radius: 4px;")

        self.btn_voice_mode = QPushButton("[VOICE: ON]")
        self.btn_voice_mode.setToolTip("Toggle Stealth / Silent Voice Mode")
        self.btn_voice_mode.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        self.btn_voice_mode.setStyleSheet("QPushButton { color: #00ffaa; background: rgba(0,255,170,0.12); border: 1px solid rgba(0,255,170,0.35); border-radius: 4px; padding: 2px 6px; } QPushButton:hover { background: rgba(0,255,170,0.28); }")
        self.btn_voice_mode.clicked.connect(self.toggle_voice_mode)

        self.time_label = QLabel()
        self.time_label.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
        self.time_label.setStyleSheet("color: #d0d8e0;")

        btn_compact = QPushButton("▫")
        btn_compact.setToolTip("Toggle Compact Mode")
        btn_compact.setFixedSize(24, 24)
        btn_compact.setStyleSheet("QPushButton { color: #00d2ff; background: rgba(0,210,255,0.15); border: 1px solid rgba(0,210,255,0.4); border-radius: 12px; font-weight: bold; } QPushButton:hover { background: rgba(0,210,255,0.35); }")
        btn_compact.clicked.connect(self.toggle_compact_mode)

        btn_close = QPushButton("✕")
        btn_close.setToolTip("Shutdown Assistant")
        btn_close.setFixedSize(24, 24)
        btn_close.setStyleSheet("QPushButton { color: #ff5566; background: rgba(255,85,102,0.15); border: 1px solid rgba(255,85,102,0.4); border-radius: 12px; font-weight: bold; } QPushButton:hover { background: rgba(255,85,102,0.35); }")
        btn_close.clicked.connect(self.close_application)

        top_bar.addWidget(self.title_label)
        top_bar.addSpacing(8)
        top_bar.addWidget(self.badge_status)
        top_bar.addSpacing(6)
        top_bar.addWidget(self.btn_voice_mode)
        top_bar.addStretch()
        top_bar.addWidget(self.time_label)
        top_bar.addSpacing(10)
        top_bar.addWidget(btn_compact)
        top_bar.addWidget(btn_close)
        root_layout.addLayout(top_bar)

        self.middle_section = QHBoxLayout()
        self.middle_section.setSpacing(12)

        self.left_panel = QFrame()
        left_layout = QVBoxLayout(self.left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        cmd_header = QHBoxLayout()
        lbl_console = QLabel("TERMINAL FEED")
        lbl_console.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        lbl_console.setStyleSheet("color: #00d2ff; letter-spacing: 1px;")
        
        btn_clear = QPushButton("CLEAR")
        btn_clear.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        btn_clear.setStyleSheet("QPushButton { background: rgba(0,210,255,0.12); border: 1px solid rgba(0,210,255,0.35); border-radius: 4px; color: #00d2ff; padding: 2px 6px; } QPushButton:hover { background: rgba(0,210,255,0.3); color: #fff; }")
        btn_clear.clicked.connect(self.clear_transcript)

        cmd_header.addWidget(lbl_console)
        cmd_header.addStretch()
        cmd_header.addWidget(btn_clear)
        left_layout.addLayout(cmd_header)

        self.console_edit = QTextEdit()
        self.console_edit.setReadOnly(True)
        left_layout.addWidget(self.console_edit, stretch=4)

        input_bar = QHBoxLayout()
        input_bar.setSpacing(6)

        self.lbl_prompt = QLabel("Aimz:~$")
        self.lbl_prompt.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        self.lbl_prompt.setStyleSheet("color: #60c5ba;")

        self.cmd_input = CyberCommandLine()
        self.cmd_input.setPlaceholderText("Type a prompt or execute command...")
        self.cmd_input.returnPressed.connect(self.handle_typed_command)

        input_bar.addWidget(self.lbl_prompt)
        input_bar.addWidget(self.cmd_input)
        left_layout.addLayout(input_bar)

        self.middle_section.addWidget(self.left_panel, stretch=2)

        center_box = QVBoxLayout()
        center_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.reactor = NeuralSynapseSwarm()
        self.reactor.clicked_mute.connect(self.toggle_mute)
        center_box.addWidget(self.reactor, alignment=Qt.AlignmentFlag.AlignCenter)

        self.status_label = QLabel("SYSTEM STANDBY")
        self.status_label.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
        self.status_label.setStyleSheet("color: #00d2ff; letter-spacing: 3px;")
        center_box.addWidget(self.status_label, alignment=Qt.AlignmentFlag.AlignCenter)
        self.middle_section.addLayout(center_box, stretch=2)

        self.right_panel = QFrame()
        right_layout = QVBoxLayout(self.right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        self.hex_stream = CyberMemoryStreamPanel()
        right_layout.addWidget(self.hex_stream, stretch=4)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        self.btn_desk = CyberChamferButton("⛶ DESKTOP")
        self.btn_desk.clicked.connect(minimize_all_windows)

        self.btn_lock = CyberChamferButton("🔒 LOCK")
        self.btn_lock.clicked.connect(lock_workstation)

        btn_row.addWidget(self.btn_desk)
        btn_row.addWidget(self.btn_lock)
        right_layout.addLayout(btn_row)

        self.middle_section.addWidget(self.right_panel, stretch=2)
        root_layout.addLayout(self.middle_section, stretch=3)

        self.bottom_panel = QFrame()
        bottom_layout = QHBoxLayout(self.bottom_panel)
        bottom_layout.setContentsMargins(0, 2, 0, 0)
        bottom_layout.setSpacing(8)
        bottom_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.graph_cpu = CyberSparklineGraph("CPU HISTORY")
        self.graph_ram = CyberSparklineGraph("RAM HISTORY")
        self.graph_disk = CyberSparklineGraph("DISK HISTORY")
        self.graph_net_up = CyberSparklineGraph("NET UP")
        self.graph_net_down = CyberSparklineGraph("NET DOWN")
        self.graph_uptime = CyberSparklineGraph("UPTIME")

        bottom_layout.addWidget(self.graph_cpu)
        bottom_layout.addWidget(self.graph_ram)
        bottom_layout.addWidget(self.graph_disk)
        bottom_layout.addWidget(self.graph_net_up)
        bottom_layout.addWidget(self.graph_net_down)
        bottom_layout.addWidget(self.graph_uptime)
        root_layout.addWidget(self.bottom_panel, stretch=1)

        self.setCentralWidget(self.main_container)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            self.is_dragging = True
            event.accept()

    def mouseMoveEvent(self, event):
        w = self.width()
        h = self.height()
        norm_x = (event.position().x() - (w / 2)) / (w / 2)
        norm_y = (event.position().y() - (h / 2)) / (h / 2)
        self.reactor.set_parallax(norm_x, norm_y)

        if event.buttons() == Qt.MouseButton.LeftButton and self.is_dragging:
            self.move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_dragging = False
            event.accept()

    def check_typing_status(self):
        global IS_USER_TYPING, LAST_TYPING_TIME
        if IS_USER_TYPING and (time.time() - LAST_TYPING_TIME > 1.2):
            IS_USER_TYPING = False

    def toggle_voice_mode(self):
        self.worker.voice_silent_mode = not self.worker.voice_silent_mode
        if self.worker.voice_silent_mode:
            self.btn_voice_mode.setText("[VOICE: OFF]")
            self.btn_voice_mode.setStyleSheet("QPushButton { color: #ffb833; background: rgba(255,184,51,0.12); border: 1px solid rgba(255,184,51,0.35); border-radius: 4px; padding: 2px 6px; } QPushButton:hover { background: rgba(255,184,51,0.28); }")
            self.console_edit.append('<span style="color: #ffb833; font-family: \'Consolas\', monospace;">// Stealth Mode: Voice playback muted (text-only).</span>')
        else:
            self.btn_voice_mode.setText("[VOICE: ON]")
            self.btn_voice_mode.setStyleSheet("QPushButton { color: #00ffaa; background: rgba(0,255,170,0.12); border: 1px solid rgba(0,255,170,0.35); border-radius: 4px; padding: 2px 6px; } QPushButton:hover { background: rgba(0,255,170,0.28); }")
            self.console_edit.append('<span style="color: #00ff8c; font-family: \'Consolas\', monospace;">// Standard Mode: Audio voice feedback active.</span>')

    def update_busy_ui(self, is_busy):
        if is_busy:
            self.badge_status.setText("[BUSY]")
            self.badge_status.setStyleSheet("color: #ffb833; background: rgba(255,184,51,0.15); padding: 2px 6px; border-radius: 4px;")
        else:
            self.badge_status.setText("[SYS_OK]")
            self.badge_status.setStyleSheet("color: #00ffaa; background: rgba(0,255,170,0.12); padding: 2px 6px; border-radius: 4px;")

    def handle_typed_command(self):
        query = self.cmd_input.text().strip()
        if not query:
            return
            
        if self.worker.is_busy:
            self.console_edit.append('<span style="color: #ff5566; font-family: \'Consolas\', monospace;">// System is currently processing another query. Please wait.</span>')
            return

        self.cmd_input.add_to_history(query)
        self.cmd_input.clear()
        self.worker.submit_text_query(query)

    def update_status(self, status):
        self.status_label.setText(f"SYSTEM {status}")
        self.reactor.set_state(status)
        self.hex_stream.set_state(status)

        palette = {
            "STANDBY": QColor(0, 210, 255),
            "LISTENING": QColor(0, 255, 170),
            "PROCESSING": QColor(255, 190, 0),
            "SPEAKING": QColor(0, 160, 255),
            "MUTED": QColor(130, 140, 150)
        }
        self.current_theme = palette.get(status, palette["STANDBY"])
        hex_c = self.current_theme.name()

        self.title_label.setStyleSheet(f"color: {hex_c}; letter-spacing: 2px;")
        self.status_label.setStyleSheet(f"color: {hex_c}; letter-spacing: 3px;")

        self.main_container.setStyleSheet(f"""
            QWidget#Container {{
                background-color: rgba(6, 12, 20, 0.94);
                border: 1.5px solid {hex_c}88;
                border-radius: 18px;
            }}
            QFrame {{
                border: none !important;
                background: transparent !important;
            }}
            QLabel {{
                font-family: 'Consolas', monospace;
                border: none !important;
                background: transparent !important;
            }}
            QTextEdit {{
                background-color: transparent !important;
                border: none !important;
                color: {hex_c};
                font-family: 'Consolas', monospace;
                font-size: 11px;
                padding: 4px;
            }}
            QLineEdit {{
                background-color: rgba(4, 10, 18, 140);
                border: 1px solid {hex_c}66;
                border-radius: 4px;
                color: {hex_c};
                font-family: 'Consolas', monospace;
                font-size: 10px;
                padding: 4px 8px;
            }}
            QLineEdit:focus {{
                border: 1px solid {hex_c};
                background-color: rgba(4, 10, 18, 200);
            }}
        """)

        self.hex_stream.set_theme_color(self.current_theme)
        self.btn_desk.set_theme_color(self.current_theme)
        self.btn_lock.set_theme_color(self.current_theme)
        self.graph_cpu.set_theme_color(self.current_theme)
        self.graph_ram.set_theme_color(self.current_theme)
        self.graph_disk.set_theme_color(self.current_theme)
        self.graph_net_up.set_theme_color(self.current_theme)
        self.graph_net_down.set_theme_color(self.current_theme)
        self.graph_uptime.set_theme_color(self.current_theme)

    def append_user_transcript(self, text):
        safe_text = html.escape(text)
        sb = self.console_edit.verticalScrollBar()
        is_at_bottom = sb.value() >= sb.maximum() - 15

        self.console_edit.append(f'<span style="color: #60c5ba; font-family: \'Consolas\', monospace; font-weight: bold;">Aimz&gt;</span> <span style="font-family: \'Consolas\', monospace;">{safe_text}</span>')
        
        if is_at_bottom:
            sb.setValue(sb.maximum())

    def handle_stream_start(self):
        sb = self.console_edit.verticalScrollBar()
        is_at_bottom = sb.value() >= sb.maximum() - 15

        cursor = self.console_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.console_edit.setTextCursor(cursor)
        self.console_edit.insertHtml('<br><br><span style="color: #00f0ff; font-family: \'Consolas\', monospace; font-weight: bold;">E.V&gt;</span> ')
        
        if is_at_bottom:
            sb.setValue(sb.maximum())

    def handle_stream_word(self, word):
        safe_word = html.escape(word)
        sb = self.console_edit.verticalScrollBar()
        is_at_bottom = sb.value() >= sb.maximum() - 15

        cursor = self.console_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.console_edit.setTextCursor(cursor)
        self.console_edit.insertHtml(f'<span style="color: #00f0ff; font-family: \'Consolas\', monospace;">{safe_word}</span>')
        
        if is_at_bottom:
            sb.setValue(sb.maximum())

    def handle_stream_end(self):
        sb = self.console_edit.verticalScrollBar()
        is_at_bottom = sb.value() >= sb.maximum() - 15

        cursor = self.console_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.console_edit.setTextCursor(cursor)
        self.console_edit.insertHtml('<br>')
        
        if is_at_bottom:
            sb.setValue(sb.maximum())

    def clear_transcript(self):
        self.console_edit.clear()
        self.console_edit.append('<span style="color: #8899a6; font-family: \'Consolas\', monospace;">// Terminal feed cleared.</span>')

    def toggle_mute(self):
        self.worker.is_muted = not self.worker.is_muted
        if self.worker.is_muted:
            self.update_status("MUTED")
            self.console_edit.append('<span style="color: #ff5566; font-family: \'Consolas\', monospace;">// Audio input stream MUTED.</span>')
        else:
            self.update_status("STANDBY")
            self.console_edit.append('<span style="color: #00ff8c; font-family: \'Consolas\', monospace;">// Audio input stream ACTIVE.</span>')

    def update_telemetry(self):
        global LAST_NET_IO, LAST_NET_TIME, NET_UP_SPEED, NET_DOWN_SPEED
        
        now = datetime.datetime.now()
        self.time_label.setText(now.strftime("%Y-%m-%d  %H:%M:%S"))

        cpu = int(psutil.cpu_percent())
        self.graph_cpu.add_data_point(cpu, subtext=f"{cpu}%")

        ram = psutil.virtual_memory()
        used_ram = round(ram.used / (1024**3), 1)
        self.graph_ram.add_data_point(ram.percent, subtext=f"{used_ram}G")

        try:
            disk = psutil.disk_usage('/')
            self.graph_disk.add_data_point(disk.percent, subtext=f"{int(disk.percent)}%")
        except Exception:
            pass

        cur_net = psutil.net_io_counters()
        cur_time = time.time()
        dt = max(1e-5, cur_time - LAST_NET_TIME)
        
        NET_UP_SPEED = round((cur_net.bytes_sent - LAST_NET_IO.bytes_sent) / 1024 / dt, 1)
        NET_DOWN_SPEED = round((cur_net.bytes_recv - LAST_NET_IO.bytes_recv) / 1024 / dt, 1)
        
        LAST_NET_IO = cur_net
        LAST_NET_TIME = cur_time
        
        self.graph_net_up.add_data_point(min(100.0, (NET_UP_SPEED / 1500.0) * 100), subtext=f"{int(NET_UP_SPEED)}K")
        self.graph_net_down.add_data_point(min(100.0, (NET_DOWN_SPEED / 3000.0) * 100), subtext=f"{int(NET_DOWN_SPEED)}K")

        uptime_sec = int(time.time() - START_TIME)
        hrs, rem = divmod(uptime_sec, 3600)
        mins, _ = divmod(rem, 60)
        self.graph_uptime.add_data_point(min(100, (uptime_sec / 86400) * 100), subtext=f"{hrs}h {mins}m")

    def toggle_compact_mode(self):
        self.is_compact = not self.is_compact
        if self.is_compact:
            self.left_panel.hide()
            self.right_panel.hide()
            self.bottom_panel.hide()
            self.setFixedSize(360, 420)
        else:
            self.left_panel.show()
            self.right_panel.show()
            self.bottom_panel.show()
            self.setFixedSize(1060, 610)

    def close_application(self):
        self.worker.is_running = False
        self.close()
        sys.exit(0)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    hud = JarvisWidescreenHUD()
    hud.show()
    sys.exit(app.exec())