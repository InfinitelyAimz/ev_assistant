import os
import sys
import json
import re
import urllib.parse
import platform
import webbrowser
import subprocess
import time
import datetime
import ctypes
import math
import warnings

# Suppress deprecation and cache warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Anchor all operations to the script's exact directory
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
    QHBoxLayout, QLabel, QFrame, QPushButton, QProgressBar, QGridLayout
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QPoint
from PyQt6.QtGui import (
    QColor, QFont, QPainter, QBrush, QPen, QRadialGradient, 
    QLinearGradient, QPolygonF
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

# Global audio visualizer energy levels
CURRENT_MIC_RMS = 0.0
START_TIME = time.time()
LAST_NET_IO = psutil.net_io_counters()
LAST_NET_TIME = time.time()

# -------------------------------------------------------------
# 1. VOICE ENGINE (PIPER TTS & WHISPER STT)
# -------------------------------------------------------------
piper_dir = os.path.join(BASE_DIR, "piper")
piper_exe = os.path.join(piper_dir, "piper.exe")
voice_model = os.path.join(piper_dir, "jarvis-medium.onnx")
json_config = os.path.join(piper_dir, "jarvis-medium.onnx.json")
temp_wav_path = os.path.join(BASE_DIR, "temp_input.wav")

def speak(text, worker_ref=None):
    """Synthesizes audio with dynamic visualizer energy streaming."""
    global CURRENT_MIC_RMS
    if not text or not text.strip():
        return

    print(f"\n[E.V.]: {text.strip()}")
    
    if os.path.exists(piper_exe) and os.path.exists(voice_model):
        try:
            process = subprocess.Popen(
                [piper_exe, "-m", voice_model, "-c", json_config, "--output_raw"],
                cwd=piper_dir,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL
            )
            raw_pcm_data, _ = process.communicate(input=text.encode('utf-8'))

            if raw_pcm_data and len(raw_pcm_data) > 0:
                audio_array = np.frombuffer(raw_pcm_data, dtype=np.int16)
                sd.play(audio_array, samplerate=16000)
                chunk_len = 1024
                for i in range(0, len(audio_array), chunk_len):
                    chunk = audio_array[i:i+chunk_len]
                    if len(chunk) > 0:
                        CURRENT_MIC_RMS = float(np.sqrt(np.mean(chunk.astype(np.float32)**2)))
                    sd.sleep(int((len(chunk) / 16000) * 1000))
                sd.wait()
                CURRENT_MIC_RMS = 0.0
                return
        except Exception as e:
            print(f"[Piper Execution Error]: {e}")

    try:
        import pyttsx3
        engine = pyttsx3.init('sapi5')
        engine.setProperty('rate', 160)
        engine.say(text)
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
    "E.V., Aimz, Sir, brightness, volume, mute, unmute, battery, "
    "system specs, RAM, CPU, weather, search, look up, price, Bitcoin, Ethereum, "
    "Solana, dollar, rand, euro, pound, exchange rate, richest, net worth, YouTube, "
    "YouTube Music, play, pause, open, VS Code, Spotify, Discord, Explorer, Task Manager."
)

def listen(silence_limit=0.6, threshold=450):
    global CURRENT_MIC_RMS
    sample_rate = 16000
    chunk_size = 1024
    
    audio_chunks = []
    silent_chunks = 0
    max_silent_chunks = int(silence_limit * (sample_rate / chunk_size))
    has_spoken = False
    
    with sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16') as stream:
        while True:
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
    wavfile.write(temp_wav_path, sample_rate, audio_data)
            
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

# -------------------------------------------------------------
# 2. HARDWARE, SYSTEM & MEDIA UTILITIES
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
        return f"Opening {app_name}, Aimz."
    except Exception as e:
        return f"Unable to launch {app_name}: {str(e)}"

def lock_workstation():
    ctypes.windll.user32.LockWorkStation()
    return "Workstation locked, Aimz."

def minimize_all_windows():
    VK_LWIN = 0x5B
    VK_D = 0x44
    KEYEVENTF_KEYUP = 0x0002
    ctypes.windll.user32.keybd_event(VK_LWIN, 0, 0, 0)
    ctypes.windll.user32.keybd_event(VK_D, 0, 0, 0)
    ctypes.windll.user32.keybd_event(VK_D, 0, KEYEVENTF_KEYUP, 0)
    ctypes.windll.user32.keybd_event(VK_LWIN, 0, KEYEVENTF_KEYUP, 0)
    return "Showing desktop, Aimz."

def get_battery_status():
    battery = psutil.sensors_battery()
    if battery is None:
        return "Battery telemetry is currently unavailable, Aimz."
    percent = battery.percent
    plugged = "plugged into AC power" if battery.power_plugged else "running on battery"
    return f"Battery is at {percent}%, {plugged}, Aimz."

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
        return "Opening your default browser now, Aimz."
    except Exception as e:
        return f"Unable to launch browser: {str(e)}"

def play_youtube_music(query: str = ""):
    try:
        clean_q = query.strip()
        if not clean_q or clean_q in ["youtube music", "yt music", "music", "songs"]:
            webbrowser.open("https://music.youtube.com")
            return "Opening YouTube Music, Aimz."

        if ytmusic_client:
            try:
                results = ytmusic_client.search(clean_q, filter="songs")
                if results and "videoId" in results[0]:
                    vid = results[0]["videoId"]
                    title = results[0].get("title", clean_q)
                    webbrowser.open(f"https://music.youtube.com/watch?v={vid}")
                    return f"Playing {title} on YouTube Music now, Aimz."
            except Exception:
                pass

        encoded_query = urllib.parse.quote_plus(clean_q)
        html = requests.get(f"https://www.youtube.com/results?search_query={encoded_query}", timeout=4).text
        video_ids = re.findall(r"watch\?v=(\S{11})", html)
        if video_ids:
            webbrowser.open(f"https://music.youtube.com/watch?v={video_ids[0]}")
            return f"Playing {clean_q} on YouTube Music now, Aimz."
        
        webbrowser.open(f"https://music.youtube.com/search?q={encoded_query}")
        return f"Searching YouTube Music for {clean_q}, Aimz."
    except Exception as e:
        return f"Unable to play track: {str(e)}"

def play_youtube_video(query: str = ""):
    try:
        clean_q = query.strip()
        if not clean_q or clean_q in ["youtube", "yt", "videos"]:
            webbrowser.open("https://www.youtube.com")
            return "Opening YouTube, Aimz."

        encoded_query = urllib.parse.quote_plus(clean_q)
        html = requests.get(f"https://www.youtube.com/results?search_query={encoded_query}", timeout=4).text
        video_ids = re.findall(r"watch\?v=(\S{11})", html)
        if video_ids:
            webbrowser.open(f"https://www.youtube.com/watch?v={video_ids[0]}")
            return f"Playing video for {clean_q} on YouTube, Aimz."
        
        webbrowser.open(f"https://www.youtube.com/results?search_query={encoded_query}")
        return f"Searching YouTube for {clean_q}, Aimz."
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
        return f"Volume adjusted to {clamped_level}%, Aimz."
    except Exception as e:
        return f"Volume adjustment failed: {str(e)}"

def set_brightness(level: int = 100):
    try:
        clamped_level = max(0, min(100, int(level)))
        try:
            sbc.set_brightness(clamped_level)
        except Exception:
            sbc.set_brightness(clamped_level, display=0)
        return f"Brightness adjusted to {clamped_level}%, Aimz."
    except Exception as e:
        return f"Brightness adjustment failed: {str(e)}"

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
    return "Media command executed, Aimz."

def get_live_weather(location=""):
    try:
        query_loc = location.strip() if location else ""
        url = f"https://wttr.in/{query_loc}?format=%C:+%t+(feels+like+%f)"
        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            return f"Weather report: {response.text.strip()}, Aimz."
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
            "xrp": "ripple"
        }
        coin_id = alias_map.get(coin_clean, coin_clean)
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
        res = requests.get(url, timeout=3).json()
        if coin_id in res and "usd" in res[coin_id]:
            price = res[coin_id]["usd"]
            return f"{coin_id.capitalize()} is currently trading at ${price:,.2f} USD, Aimz."
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
            return f"The current exchange rate for 1 {base_clean} is {rate:,.2f} {target_clean}, Aimz."
    except Exception:
        pass
    return None

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
        q = re.sub(r'^(play|search for|search|open)\s+(song|track|music)?\s*', '', clean)
        q = re.sub(r'\s+(on|in)?\s*(youtube music|yt music)$', '', q).strip()
        return play_youtube_music(q)

    if "youtube" in clean or "yt" in clean or clean.startswith("play video"):
        q = re.sub(r'^(play|search for|search|look up|open)\s+(video)?\s*', '', clean)
        q = re.sub(r'\s+(on|in)?\s*(youtube|yt)$', '', q).strip()
        return play_youtube_video(q)

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
    ai_response_generated = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.is_running = True
        self.is_muted = False

    def run(self):
        self.status_changed.emit("SPEAKING")
        self.ai_response_generated.emit("Visual diagnostics online. Tactical HUD initialized.")
        speak("Visual diagnostics online. Tactical HUD initialized.", self)
        self.status_changed.emit("STANDBY")

        system_prompt = {
            "role": "system", 
            "content": (
                "You are E.V., an articulate AI assistant modeled after J.A.R.V.I.S. "
                "Address the user as Aimz or Sir. "
                "Keep answers strictly to 1 concise sentence. "
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
                if self.is_muted:
                    time.sleep(0.2)
                    continue

                self.status_changed.emit("LISTENING")
                user_input = listen(silence_limit=0.6, threshold=450)

                if not user_input or len(user_input.strip()) < 2:
                    self.status_changed.emit("STANDBY")
                    continue

                self.user_speech_detected.emit(user_input)
                clean_input = user_input.lower().replace(".", "").strip()

                if any(w in clean_input for w in ["shutdown", "shut down", "goodbye", "exit", "stop"]):
                    self.status_changed.emit("SPEAKING")
                    self.ai_response_generated.emit("Systems going offline. Goodbye, Aimz.")
                    speak("Systems going offline. Goodbye, Aimz.", self)
                    self.is_running = False
                    break

                self.status_changed.emit("PROCESSING")
                fast_reply = handle_direct_commands(clean_input)
                
                if fast_reply:
                    self.status_changed.emit("SPEAKING")
                    self.ai_response_generated.emit(fast_reply)
                    messages.append({"role": "user", "content": user_input})
                    messages.append({"role": "assistant", "content": fast_reply})
                    speak(fast_reply, self)
                    self.status_changed.emit("STANDBY")
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
                self.ai_response_generated.emit(reply)
                messages.append({"role": "assistant", "content": reply})
                speak(reply, self)
                self.status_changed.emit("STANDBY")

            except Exception as e:
                print(f"[Loop Exception]: {e}")
                self.status_changed.emit("STANDBY")

# -------------------------------------------------------------
# 4. ADVANCED J.A.R.V.I.S. ARC REACTOR HUD DIAL
# -------------------------------------------------------------
class JarvisArcReactorDial(QWidget):
    """Multi-ring sci-fi cybernetic Arc Reactor dial with animated rings & visualizer."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(310, 310)
        self.angle_outer = 0.0
        self.angle_mid = 0.0
        self.angle_inner = 0.0
        self.state = "STANDBY"
        
        self.bars = 36
        self.audio_levels = [0.0] * self.bars
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(25)

    def set_state(self, state):
        self.state = state

    def update_animation(self):
        global CURRENT_MIC_RMS
        speed_mult = 3.5 if self.state == "PROCESSING" else (2.0 if self.state == "SPEAKING" else 1.0)
        self.angle_outer = (self.angle_outer + 0.8 * speed_mult) % 360
        self.angle_mid = (self.angle_mid - 1.4 * speed_mult) % 360
        self.angle_inner = (self.angle_inner + 2.2 * speed_mult) % 360
        
        target_amp = min(1.0, CURRENT_MIC_RMS / 2400.0)
        if self.state in ["LISTENING", "SPEAKING"]:
            target_amp = max(0.18, target_amp)
        else:
            target_amp = 0.04

        for i in range(self.bars):
            var = math.sin((self.angle_outer * 0.08) + (i * 0.45)) * 0.22
            val = max(0.04, target_amp + var)
            self.audio_levels[i] += (val - self.audio_levels[i]) * 0.35
            
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        cx, cy = self.width() / 2, self.height() / 2

        # State color mappings
        colors = {
            "STANDBY": (QColor(0, 215, 255), QColor(0, 215, 255, 30)),
            "LISTENING": (QColor(0, 255, 170), QColor(0, 255, 170, 60)),
            "PROCESSING": (QColor(255, 185, 0), QColor(255, 185, 0, 60)),
            "SPEAKING": (QColor(0, 165, 255), QColor(0, 165, 255, 75)),
            "MUTED": (QColor(255, 60, 90), QColor(255, 60, 90, 50))
        }
        primary, glow = colors.get(self.state, colors["STANDBY"])

        # 1. Background Radial Glow
        bg_grad = QRadialGradient(cx, cy, 145)
        bg_grad.setColorAt(0.0, glow)
        bg_grad.setColorAt(0.7, QColor(0, 0, 0, 100))
        bg_grad.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.setBrush(QBrush(bg_grad))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(0, 0, self.width(), self.height())

        # 2. Outer Segmented Tick Ring (Image 1 style)
        painter.save()
        painter.translate(cx, cy)
        painter.rotate(self.angle_outer * 0.4)
        num_ticks = 48
        for i in range(num_ticks):
            is_major = (i % 4 == 0)
            t_len = 8 if is_major else 4
            t_pen = QPen(primary if is_major else QColor(0, 215, 255, 100), 1.8 if is_major else 1.0)
            painter.setPen(t_pen)
            painter.drawLine(0, 142, 0, 142 - t_len)
            painter.rotate(360 / num_ticks)
        painter.restore()

        # 3. Outer Arcs & Crosshair Reticles
        pen = QPen(primary, 2, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawArc(int(cx - 128), int(cy - 128), 256, 256, int(self.angle_outer * 16), 110 * 16)
        painter.drawArc(int(cx - 128), int(cy - 128), 256, 256, int((self.angle_outer + 180) * 16), 110 * 16)

        # 4. Radial Audio Visualizer Spikes
        painter.save()
        painter.translate(cx, cy)
        r_inner = 88
        for i in range(self.bars):
            rot = (360.0 / self.bars) * i + self.angle_mid
            painter.save()
            painter.rotate(rot)
            bar_len = 4 + (self.audio_levels[i] * 34)
            pen = QPen(primary, 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawLine(0, int(r_inner), 0, int(r_inner + bar_len))
            painter.restore()
        painter.restore()

        # 5. Mid Segmented Chevron/Hazard Ring
        pen = QPen(QColor(0, 215, 255, 160), 1.5, Qt.PenStyle.DotLine)
        painter.setPen(pen)
        painter.drawEllipse(int(cx - 82), int(cy - 82), 164, 164)

        # 6. Inner Fast Counter-Rotating Arcs
        pen_inner = QPen(primary, 2.5, Qt.PenStyle.SolidLine)
        painter.setPen(pen_inner)
        painter.drawArc(int(cx - 62), int(cy - 62), 124, 124, int(self.angle_inner * 16), 75 * 16)
        painter.drawArc(int(cx - 62), int(cy - 62), 124, 124, int((self.angle_inner + 180) * 16), 75 * 16)

        # 7. Central Core Reactor Badge
        core_grad = QRadialGradient(cx, cy, 38)
        core_grad.setColorAt(0.0, QColor(255, 255, 255, 255))
        core_grad.setColorAt(0.4, primary)
        core_grad.setColorAt(1.0, QColor(4, 10, 18, 230))
        painter.setBrush(QBrush(core_grad))
        painter.setPen(QPen(primary, 1.5))
        painter.drawEllipse(int(cx - 38), int(cy - 38), 76, 76)

        # 8. Center Label / State Code
        painter.setPen(QPen(QColor(255, 255, 255, 220)))
        painter.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
        display_code = "E.V." if self.state == "STANDBY" else self.state[:4]
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, display_code)


# -------------------------------------------------------------
# 5. FULL TACTICAL J.A.R.V.I.S. HUD DASHBOARD
# -------------------------------------------------------------
class JarvisHUD(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("J.A.R.V.I.S. Tactical Neural HUD")
        self.setFixedSize(920, 580)
        
        # Transparent Frameless Window
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self.drag_position = QPoint()
        self.is_compact = False
        
        # Typewriter text streaming state
        self.typing_target_text = ""
        self.typing_index = 0
        self.typing_timer = QTimer(self)
        self.typing_timer.timeout.connect(self.stream_text_character)

        self.init_ui()

        # Start Background Voice Thread
        self.worker = AssistantWorker()
        self.worker.status_changed.connect(self.update_status)
        self.worker.user_speech_detected.connect(self.update_user_text)
        self.worker.ai_response_generated.connect(self.trigger_ai_typing)
        self.worker.start()

        # Real-time System Telemetry Timer
        self.stats_timer = QTimer(self)
        self.stats_timer.timeout.connect(self.update_telemetry)
        self.stats_timer.start(1000)
        self.update_telemetry()

    def init_ui(self):
        self.main_container = QWidget(self)
        self.main_container.setObjectName("Container")
        self.main_container.setStyleSheet("""
            QWidget#Container {
                background-color: rgba(6, 11, 18, 0.94);
                border: 1.5px solid rgba(0, 215, 255, 0.5);
                border-radius: 18px;
            }
            QLabel {
                font-family: 'Consolas', 'Segoe UI', monospace;
            }
            QProgressBar {
                border: 1px solid rgba(0, 215, 255, 0.35);
                border-radius: 3px;
                background-color: rgba(4, 8, 14, 0.9);
                text-align: right;
                padding-right: 4px;
                color: #00f0ff;
                font-size: 9px;
                font-family: 'Consolas';
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #005588, stop:1 #00f0ff);
                border-radius: 2px;
            }
        """)

        root_layout = QVBoxLayout(self.main_container)
        root_layout.setContentsMargins(20, 16, 20, 16)
        root_layout.setSpacing(10)

        # 1. Top HUD Header
        top_bar = QHBoxLayout()
        title_label = QLabel("MARK-VII // E.V. TACTICAL SYSTEM MONITOR")
        title_label.setFont(QFont("Consolas", 11, QFont.Weight.Bold))
        title_label.setStyleSheet("color: #00f0ff; letter-spacing: 2.5px;")
        
        self.time_label = QLabel()
        self.time_label.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
        self.time_label.setStyleSheet("color: #00d7ff;")

        btn_compact = QPushButton("▫")
        btn_compact.setToolTip("Toggle Compact Mode")
        btn_compact.setFixedSize(26, 26)
        btn_compact.setStyleSheet("QPushButton { color: #00d7ff; background: rgba(0,215,255,0.12); border: 1px solid rgba(0,215,255,0.4); border-radius: 13px; font-weight: bold; } QPushButton:hover { background: rgba(0,215,255,0.3); }")
        btn_compact.clicked.connect(self.toggle_compact_mode)

        btn_close = QPushButton("✕")
        btn_close.setToolTip("Shutdown Assistant")
        btn_close.setFixedSize(26, 26)
        btn_close.setStyleSheet("QPushButton { color: #ff5566; background: rgba(255,85,102,0.12); border: 1px solid rgba(255,85,102,0.4); border-radius: 13px; font-weight: bold; } QPushButton:hover { background: rgba(255,85,102,0.3); }")
        btn_close.clicked.connect(self.close_application)

        top_bar.addWidget(title_label)
        top_bar.addStretch()
        top_bar.addWidget(self.time_label)
        top_bar.addSpacing(12)
        top_bar.addWidget(btn_compact)
        top_bar.addWidget(btn_close)
        root_layout.addLayout(top_bar)

        # 2. Main Middle Section (Arc Reactor Center + Telemetry Side-Panels)
        self.middle_section = QHBoxLayout()
        self.middle_section.setSpacing(16)

        # LEFT PANEL: System Monitor & Network Gauges (Image 2 style)
        self.left_panel = QFrame()
        self.left_panel.setStyleSheet("QFrame { background: rgba(3, 8, 14, 0.75); border: 1px solid rgba(0,215,255,0.25); border-radius: 10px; padding: 8px; }")
        left_layout = QVBoxLayout(self.left_panel)
        left_layout.setSpacing(6)

        panel_title = QLabel("SYSTEM METRICS")
        panel_title.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
        panel_title.setStyleSheet("color: #00f0ff; letter-spacing: 1.5px;")
        left_layout.addWidget(panel_title)

        # CPU Progress
        self.lbl_cpu_text = QLabel("CPU Usage: 0%")
        self.lbl_cpu_text.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        self.lbl_cpu_text.setStyleSheet("color: #00d7ff;")
        self.bar_cpu = QProgressBar()
        self.bar_cpu.setFixedHeight(12)
        left_layout.addWidget(self.lbl_cpu_text)
        left_layout.addWidget(self.bar_cpu)

        # RAM Progress
        self.lbl_ram_text = QLabel("RAM: 0.0G / 0.0G (0%)")
        self.lbl_ram_text.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        self.lbl_ram_text.setStyleSheet("color: #00d7ff;")
        self.bar_ram = QProgressBar()
        self.bar_ram.setFixedHeight(12)
        left_layout.addWidget(self.lbl_ram_text)
        left_layout.addWidget(self.bar_ram)

        # Disk Usage Progress
        self.lbl_disk_text = QLabel("Disk Usage: 0.0G / 0.0G")
        self.lbl_disk_text.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        self.lbl_disk_text.setStyleSheet("color: #00d7ff;")
        self.bar_disk = QProgressBar()
        self.bar_disk.setFixedHeight(12)
        left_layout.addWidget(self.lbl_disk_text)
        left_layout.addWidget(self.bar_disk)

        # Network / Process Readout
        self.lbl_net_up = QLabel("Network Up: 0 KB/s")
        self.lbl_net_up.setFont(QFont("Consolas", 8))
        self.lbl_net_up.setStyleSheet("color: #8bbcdb;")
        
        self.lbl_net_down = QLabel("Network Down: 0 KB/s")
        self.lbl_net_down.setFont(QFont("Consolas", 8))
        self.lbl_net_down.setStyleSheet("color: #8bbcdb;")

        self.lbl_uptime = QLabel("Uptime: 0h 0m")
        self.lbl_uptime.setFont(QFont("Consolas", 8))
        self.lbl_uptime.setStyleSheet("color: #8bbcdb;")

        self.lbl_processes = QLabel("Processes: 0")
        self.lbl_processes.setFont(QFont("Consolas", 8))
        self.lbl_processes.setStyleSheet("color: #8bbcdb;")

        left_layout.addWidget(self.lbl_net_up)
        left_layout.addWidget(self.lbl_net_down)
        left_layout.addWidget(self.lbl_uptime)
        left_layout.addWidget(self.lbl_processes)
        self.middle_section.addWidget(self.left_panel, stretch=1)

        # CENTER: Arc Reactor Dial Core
        center_box = QVBoxLayout()
        center_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.reactor = JarvisArcReactorDial()
        center_box.addWidget(self.reactor, alignment=Qt.AlignmentFlag.AlignCenter)

        self.status_label = QLabel("SYSTEM STANDBY")
        self.status_label.setFont(QFont("Consolas", 11, QFont.Weight.Bold))
        self.status_label.setStyleSheet("color: #00f0ff; letter-spacing: 3px;")
        center_box.addWidget(self.status_label, alignment=Qt.AlignmentFlag.AlignCenter)
        self.middle_section.addLayout(center_box, stretch=1)

        # RIGHT PANEL: Direct Action Dock & Status
        self.right_panel = QFrame()
        self.right_panel.setStyleSheet("QFrame { background: rgba(3, 8, 14, 0.75); border: 1px solid rgba(0,215,255,0.25); border-radius: 10px; padding: 8px; }")
        right_layout = QVBoxLayout(self.right_panel)
        right_layout.setSpacing(8)

        action_title = QLabel("TACTICAL DOCK")
        action_title.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
        action_title.setStyleSheet("color: #00f0ff; letter-spacing: 1.5px;")
        right_layout.addWidget(action_title)

        btn_style = """
            QPushButton {
                background: rgba(0, 215, 255, 0.1);
                border: 1px solid rgba(0, 215, 255, 0.35);
                border-radius: 6px;
                color: #a0d4f2;
                font-size: 10px;
                font-family: 'Consolas';
                font-weight: bold;
                padding: 6px 10px;
            }
            QPushButton:hover {
                background: rgba(0, 215, 255, 0.28);
                color: #ffffff;
            }
        """
        self.btn_mute = QPushButton("🎙 MUTE INPUT")
        self.btn_mute.setStyleSheet(btn_style)
        self.btn_mute.clicked.connect(self.toggle_mute)

        btn_clear = QPushButton("⟲ CLEAR TRANSCRIPT")
        btn_clear.setStyleSheet(btn_style)
        btn_clear.clicked.connect(self.clear_transcript)

        btn_media = QPushButton("⏯ MEDIA PLAY/PAUSE")
        btn_media.setStyleSheet(btn_style)
        btn_media.clicked.connect(lambda: media_control("play_pause"))

        btn_desk = QPushButton("⛶ SHOW DESKTOP")
        btn_desk.setStyleSheet(btn_style)
        btn_desk.clicked.connect(minimize_all_windows)

        btn_lock = QPushButton("🔒 LOCK SYSTEM")
        btn_lock.setStyleSheet(btn_style)
        btn_lock.clicked.connect(lock_workstation)

        right_layout.addWidget(self.btn_mute)
        right_layout.addWidget(btn_clear)
        right_layout.addWidget(btn_media)
        right_layout.addWidget(btn_desk)
        right_layout.addWidget(btn_lock)
        right_layout.addStretch()

        self.middle_section.addWidget(self.right_panel, stretch=1)
        root_layout.addLayout(self.middle_section)

        # 3. Bottom Transcripts Dialogue Section
        self.dialogue_frame = QFrame()
        self.dialogue_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(3, 8, 14, 0.88);
                border: 1px solid rgba(0, 215, 255, 0.3);
                border-radius: 10px;
                padding: 8px;
            }
        """)
        d_layout = QVBoxLayout(self.dialogue_frame)
        d_layout.setSpacing(4)

        self.user_label = QLabel("Aimz: [Awaiting Input...]")
        self.user_label.setFont(QFont("Segoe UI", 9))
        self.user_label.setStyleSheet("color: #7f99b2;")
        self.user_label.setWordWrap(True)

        self.ai_label = QLabel("E.V.: Online and standing by.")
        self.ai_label.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        self.ai_label.setStyleSheet("color: #00f0ff;")
        self.ai_label.setWordWrap(True)

        d_layout.addWidget(self.user_label)
        d_layout.addWidget(self.ai_label)
        root_layout.addWidget(self.dialogue_frame)

        self.setCentralWidget(self.main_container)

    # ---------------------------------------------------------
    # HUD Controller Methods
    # ---------------------------------------------------------
    def update_status(self, status):
        self.status_label.setText(f"SYSTEM {status}")
        self.reactor.set_state(status)

    def update_user_text(self, text):
        self.user_label.setText(f"Aimz: {text}")

    def trigger_ai_typing(self, text):
        self.typing_target_text = text
        self.typing_index = 0
        self.ai_label.setText("E.V.: ")
        self.typing_timer.start(16)

    def stream_text_character(self):
        if self.typing_index < len(self.typing_target_text):
            current = self.ai_label.text() + self.typing_target_text[self.typing_index]
            self.ai_label.setText(current)
            self.typing_index += 1
        else:
            self.typing_timer.stop()

    def update_telemetry(self):
        global LAST_NET_IO, LAST_NET_TIME
        
        # Clock & Date
        now = datetime.datetime.now()
        self.time_label.setText(now.strftime("%Y-%m-%d  %H:%M:%S"))

        # CPU
        cpu = int(psutil.cpu_percent())
        self.bar_cpu.setValue(cpu)
        self.lbl_cpu_text.setText(f"CPU Usage: {cpu}%")

        # RAM
        ram = psutil.virtual_memory()
        used_ram = round(ram.used / (1024**3), 2)
        total_ram = round(ram.total / (1024**3), 2)
        self.bar_ram.setValue(int(ram.percent))
        self.lbl_ram_text.setText(f"RAM: {used_ram}G / {total_ram}G - {int(ram.percent)}%")

        # Disk
        try:
            disk = psutil.disk_usage('/')
            used_disk = round(disk.used / (1024**3), 1)
            total_disk = round(disk.total / (1024**3), 1)
            self.bar_disk.setValue(int(disk.percent))
            self.lbl_disk_text.setText(f"Disk Usage: {used_disk}G / {total_disk}G")
        except Exception:
            pass

        # Network Traffic Speed
        cur_net = psutil.net_io_counters()
        cur_time = time.time()
        dt = max(1e-5, cur_time - LAST_NET_TIME)
        
        up_speed = round((cur_net.bytes_sent - LAST_NET_IO.bytes_sent) / 1024 / dt, 1)
        down_speed = round((cur_net.bytes_recv - LAST_NET_IO.bytes_recv) / 1024 / dt, 1)
        
        LAST_NET_IO = cur_net
        LAST_NET_TIME = cur_time
        
        self.lbl_net_up.setText(f"Network Up: {up_speed} KB/s")
        self.lbl_net_down.setText(f"Network Down: {down_speed} KB/s")

        # Uptime & Processes
        uptime_sec = int(time.time() - START_TIME)
        hrs, rem = divmod(uptime_sec, 3600)
        mins, _ = divmod(rem, 60)
        self.lbl_uptime.setText(f"Uptime: {hrs}h {mins}m")
        self.lbl_processes.setText(f"Processes: {len(psutil.pids())}")

    def toggle_mute(self):
        self.worker.is_muted = not self.worker.is_muted
        if self.worker.is_muted:
            self.btn_mute.setText("🔇 UNMUTE INPUT")
            self.btn_mute.setStyleSheet("QPushButton { background: rgba(255, 60, 90, 0.25); border: 1px solid #ff3c5a; color: #ffffff; border-radius: 6px; font-size: 10px; font-weight: bold; padding: 6px 10px; }")
            self.update_status("MUTED")
        else:
            self.btn_mute.setText("🎙 MUTE INPUT")
            self.btn_mute.setStyleSheet("QPushButton { background: rgba(0, 215, 255, 0.1); border: 1px solid rgba(0, 215, 255, 0.35); border-radius: 6px; color: #a0d4f2; font-size: 10px; font-weight: bold; padding: 6px 10px; }")
            self.update_status("STANDBY")

    def clear_transcript(self):
        self.user_label.setText("Aimz: [Awaiting Input...]")
        self.ai_label.setText("E.V.: Ready for command.")

    def toggle_compact_mode(self):
        self.is_compact = not self.is_compact
        if self.is_compact:
            self.left_panel.hide()
            self.right_panel.hide()
            self.dialogue_frame.hide()
            self.setFixedSize(360, 420)
        else:
            self.left_panel.show()
            self.right_panel.show()
            self.dialogue_frame.show()
            self.setFixedSize(920, 580)

    # Holographic Corner Targeting Reticles
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        pen = QPen(QColor(0, 240, 255, 200), 2)
        painter.setPen(pen)

        w = self.width()
        h = self.height()
        c_len = 18

        # Top-Left Reticle
        painter.drawLine(10, 10 + c_len, 10, 10)
        painter.drawLine(10, 10, 10 + c_len, 10)

        # Top-Right Reticle
        painter.drawLine(w - 10 - c_len, 10, w - 10, 10)
        painter.drawLine(w - 10, 10, w - 10, 10 + c_len)

        # Bottom-Left Reticle
        painter.drawLine(10, h - 10 - c_len, 10, h - 10)
        painter.drawLine(10, h - 10, 10 + c_len, h - 10)

        # Bottom-Right Reticle
        painter.drawLine(w - 10 - c_len, h - 10, w - 10, h - 10)
        painter.drawLine(w - 10, h - 10 - c_len, w - 10, h - 10)

    # Draggable Window Logic
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()

    def close_application(self):
        self.worker.is_running = False
        self.close()
        sys.exit(0)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    hud = JarvisHUD()
    hud.show()
    sys.exit(app.exec())