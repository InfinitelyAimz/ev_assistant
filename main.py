import os
import sys
import json
import re
import urllib.parse
import platform
import webbrowser
import subprocess
import time
import ctypes
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
    QHBoxLayout, QLabel, QFrame
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QPoint
from PyQt6.QtGui import QColor, QFont, QPainter, QBrush, QPen, QRadialGradient

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

try:
    from ytmusicapi import YTMusic
    ytmusic_client = YTMusic()
except Exception:
    ytmusic_client = None

# -------------------------------------------------------------
# 1. VOICE ENGINE (PIPER TTS & WHISPER STT)
# -------------------------------------------------------------
piper_dir = os.path.join(BASE_DIR, "piper")
piper_exe = os.path.join(piper_dir, "piper.exe")
voice_model = os.path.join(piper_dir, "jarvis-medium.onnx")
json_config = os.path.join(piper_dir, "jarvis-medium.onnx.json")
temp_wav_path = os.path.join(BASE_DIR, "temp_input.wav")

def speak(text):
    """Synthesizes and plays audio directly out loud."""
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
                sd.wait()
                return
        except Exception as e:
            print(f"[Piper Execution Error]: {e}")

    # Fallback to standard SAPI5
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
            
            if rms > threshold:
                has_spoken = True
                silent_chunks = 0
            elif has_spoken:
                silent_chunks += 1
                if silent_chunks > max_silent_chunks:
                    break
            
            if len(audio_chunks) > int(4.5 * (sample_rate / chunk_size)) and not has_spoken:
                return ""

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
        if not query:
            webbrowser.open("https://music.youtube.com")
            return "Opening YouTube Music, Aimz."

        if ytmusic_client:
            try:
                results = ytmusic_client.search(query, filter="songs")
                if results and "videoId" in results[0]:
                    vid = results[0]["videoId"]
                    title = results[0].get("title", query)
                    webbrowser.open(f"https://music.youtube.com/watch?v={vid}")
                    return f"Playing {title} on YouTube Music now, Aimz."
            except Exception:
                pass

        encoded_query = urllib.parse.quote_plus(query)
        html = requests.get(f"https://www.youtube.com/results?search_query={encoded_query}", timeout=4).text
        video_ids = re.findall(r"watch\?v=(\S{11})", html)
        
        if video_ids:
            webbrowser.open(f"https://music.youtube.com/watch?v={video_ids[0]}")
            return f"Playing {query} on YouTube Music now, Aimz."
        
        webbrowser.open(f"https://music.youtube.com/search?q={encoded_query}")
        return f"Searching YouTube Music for {query}, Aimz."
    except Exception as e:
        return f"Unable to play track: {str(e)}"

def play_youtube_video(query: str = ""):
    try:
        if not query:
            webbrowser.open("https://www.youtube.com")
            return "Opening YouTube, Aimz."

        encoded_query = urllib.parse.quote_plus(query)
        html = requests.get(f"https://www.youtube.com/results?search_query={encoded_query}", timeout=4).text
        video_ids = re.findall(r"watch\?v=(\S{11})", html)
        
        if video_ids:
            webbrowser.open(f"https://www.youtube.com/watch?v={video_ids[0]}")
            return f"Playing video for {query} on YouTube, Aimz."
        
        webbrowser.open(f"https://www.youtube.com/results?search_query={encoded_query}")
        return f"Searching YouTube for {query}, Aimz."
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

    if any(k in clean for k in ["youtube music", "yt music"]) or (clean.startswith("play") and any(w in clean for w in ["song", "track", "music"])):
        q = re.sub(r'^(play|search for|search|open)\s+(song|track|music)?\s*', '', clean)
        q = re.sub(r'\s+(on|in)?\s*(youtube music|yt music)$', '', q).strip()
        return play_youtube_music(q)

    if "youtube" in clean or "yt" in clean or clean.startswith("play video"):
        q = re.sub(r'^(play|search for|search|look up|open)\s+(video)?\s*', '', clean)
        q = re.sub(r'\s+(on|in)?\s*(youtube|yt)$', '', q).strip()
        return play_youtube_video(q)

    if any(k in clean for k in ["pause", "resume", "unpause"]):
        return media_control("play_pause")
    if any(w in clean for w in ["next track", "next song", "skip track", "skip song"]):
        return media_control("next")
    if any(w in clean for w in ["previous track", "previous song", "back song"]):
        return media_control("previous")

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
    status_changed = pyqtSignal(str)     # "STANDBY", "LISTENING", "PROCESSING", "SPEAKING"
    user_speech_detected = pyqtSignal(str)
    ai_response_generated = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.is_running = True

    def run(self):
        self.status_changed.emit("SPEAKING")
        speak("E.V. visual interface online and operational.")
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
                    speak("Systems going offline. Goodbye, Aimz.")
                    self.is_running = False
                    break

                self.status_changed.emit("PROCESSING")
                fast_reply = handle_direct_commands(clean_input)
                
                if fast_reply:
                    self.status_changed.emit("SPEAKING")
                    self.ai_response_generated.emit(fast_reply)
                    messages.append({"role": "user", "content": user_input})
                    messages.append({"role": "assistant", "content": fast_reply})
                    speak(fast_reply)
                    self.status_changed.emit("STANDBY")
                    continue

                # Live Search / LLaMA Fallback
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
                speak(reply)
                self.status_changed.emit("STANDBY")

            except Exception as e:
                print(f"[Loop Exception]: {e}")
                self.status_changed.emit("STANDBY")

# -------------------------------------------------------------
# 4. PYQT6 HUD WIDGET & VISUAL DASHBOARD
# -------------------------------------------------------------
class ArcReactorWidget(QWidget):
    """Futuristic glowing circular Arc Reactor core with state animations."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(160, 160)
        self.angle = 0
        self.state = "STANDBY"
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(30)

    def set_state(self, state):
        self.state = state
        self.update()

    def update_animation(self):
        self.angle = (self.angle + (4 if self.state == "PROCESSING" else 1.5)) % 360
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        center_x, center_y = self.width() / 2, self.height() / 2

        # State color palettes
        colors = {
            "STANDBY": (QColor(0, 210, 255, 180), QColor(0, 210, 255, 30)),
            "LISTENING": (QColor(0, 255, 170, 230), QColor(0, 255, 170, 60)),
            "PROCESSING": (QColor(255, 190, 0, 230), QColor(255, 190, 0, 60)),
            "SPEAKING": (QColor(0, 150, 255, 255), QColor(0, 150, 255, 80))
        }
        primary_color, glow_color = colors.get(self.state, colors["STANDBY"])

        # Outer Glow
        gradient = QRadialGradient(center_x, center_y, 75)
        gradient.setColorAt(0.0, glow_color)
        gradient.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(5, 5, 150, 150)

        # Outer Rotating Ring
        pen = QPen(primary_color, 2, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawArc(18, 18, 124, 124, int(self.angle * 16), 180 * 16)
        painter.drawArc(18, 18, 124, 124, int((self.angle + 180) * 16), 90 * 16)

        # Inner Solid Ring
        pen = QPen(primary_color, 1.5, Qt.PenStyle.SolidLine)
        painter.setPen(pen)
        painter.drawEllipse(34, 34, 92, 92)

        # Central Core
        core_grad = QRadialGradient(center_x, center_y, 30)
        core_grad.setColorAt(0.0, QColor(255, 255, 255, 240))
        core_grad.setColorAt(0.5, primary_color)
        core_grad.setColorAt(1.0, QColor(0, 0, 0, 150))
        painter.setBrush(QBrush(core_grad))
        painter.setPen(QPen(primary_color, 1))
        painter.drawEllipse(48, 48, 64, 64)


class JarvisHUD(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("E.V. Neural Interface")
        self.setFixedSize(560, 480)
        
        # Transparent Frameless Window
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self.drag_position = QPoint()
        self.init_ui()

        # Start Background Voice Thread
        self.worker = AssistantWorker()
        self.worker.status_changed.connect(self.update_status)
        self.worker.user_speech_detected.connect(self.update_user_text)
        self.worker.ai_response_generated.connect(self.update_ai_text)
        self.worker.start()

        # Telemetry Timer
        self.stats_timer = QTimer(self)
        self.stats_timer.timeout.connect(self.update_telemetry)
        self.stats_timer.start(2000)
        self.update_telemetry()

    def init_ui(self):
        main_widget = QWidget(self)
        main_widget.setObjectName("Container")
        main_widget.setStyleSheet("""
            QWidget#Container {
                background-color: rgba(10, 16, 26, 0.92);
                border: 1.5px solid rgba(0, 210, 255, 0.4);
                border-radius: 18px;
            }
        """)

        layout = QVBoxLayout(main_widget)
        layout.setContentsMargins(25, 20, 25, 20)
        layout.setSpacing(12)

        # Top Bar: Title & Close
        top_bar = QHBoxLayout()
        title_label = QLabel("E.V. SYSTEM HUD // MARK-IV")
        title_label.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
        title_label.setStyleSheet("color: #00d2ff; letter-spacing: 2px;")
        
        close_btn = QLabel("✕")
        close_btn.setFont(QFont("Consolas", 12, QFont.Weight.Bold))
        close_btn.setStyleSheet("color: #667788; cursor: pointer;")
        close_btn.mousePressEvent = lambda e: self.close_application()

        top_bar.addWidget(title_label)
        top_bar.addStretch()
        top_bar.addWidget(close_btn)
        layout.addLayout(top_bar)

        # Center: Arc Reactor & Status
        center_layout = QVBoxLayout()
        center_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.reactor = ArcReactorWidget()
        center_layout.addWidget(self.reactor, alignment=Qt.AlignmentFlag.AlignCenter)

        self.status_label = QLabel("SYSTEM STANDBY")
        self.status_label.setFont(QFont("Consolas", 11, QFont.Weight.Bold))
        self.status_label.setStyleSheet("color: #00d2ff; letter-spacing: 3px;")
        center_layout.addWidget(self.status_label, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addLayout(center_layout)

        # Dialogue Section
        dialogue_frame = QFrame()
        dialogue_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(4, 8, 14, 0.7);
                border: 1px solid rgba(0, 210, 255, 0.18);
                border-radius: 10px;
                padding: 6px;
            }
        """)
        d_layout = QVBoxLayout(dialogue_frame)
        d_layout.setSpacing(6)

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
        layout.addWidget(dialogue_frame)

        # Bottom Bar: Live Telemetry
        self.telemetry_label = QLabel("CPU: 0% | RAM: 0GB | BATTERY: 100%")
        self.telemetry_label.setFont(QFont("Consolas", 8))
        self.telemetry_label.setStyleSheet("color: #4c687e;")
        self.telemetry_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.telemetry_label)

        self.setCentralWidget(main_widget)

    def update_status(self, status):
        self.status_label.setText(f"SYSTEM {status}")
        self.reactor.set_state(status)

    def update_user_text(self, text):
        self.user_label.setText(f"Aimz: {text}")

    def update_ai_text(self, text):
        self.ai_label.setText(f"E.V.: {text}")

    def update_telemetry(self):
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory()
        used_ram = round(ram.used / (1024**3), 1)
        total_ram = round(ram.total / (1024**3), 1)
        
        battery = psutil.sensors_battery()
        bat_str = f"{battery.percent}%" if battery else "AC"
        self.telemetry_label.setText(f"CPU: {cpu}%  |  RAM: {used_ram}/{total_ram} GB  |  BATTERY: {bat_str}")

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