import os
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

import sounddevice as sd
import numpy as np
from scipy.io import wavfile
from faster_whisper import WhisperModel
import ollama
import psutil
import requests
import screen_brightness_control as sbc
from pycaw.pycaw import AudioUtilities

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
# 1. SETUP VOICE ENGINE (PIPER NEURAL TTS & WHISPER STT)
# -------------------------------------------------------------
piper_dir = os.path.abspath("piper")
piper_exe = os.path.join(piper_dir, "piper.exe")
voice_model = os.path.join(piper_dir, "jarvis-medium.onnx")
json_config = os.path.join(piper_dir, "jarvis-medium.onnx.json")

def speak(text):
    """Synthesizes and plays audio directly out loud."""
    if not text or not text.strip():
        return

    print(f"\n[E.V.]: {text.strip()}")
    
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
    """Listens with dynamic cutoff and prevents empty audio processing."""
    sample_rate = 16000
    chunk_size = 1024
    
    print(f"\n[Listening... Speak into your mic]")
    t_start = time.time()
    
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

    t_mic = round(time.time() - t_start, 2)
    t_whisper_start = time.time()
    
    audio_data = np.concatenate(audio_chunks, axis=0)
    wavfile.write("temp_input.wav", sample_rate, audio_data)
            
    segments, _ = stt_model.transcribe(
        "temp_input.wav", 
        language="en", 
        beam_size=1,
        best_of=1,
        vad_filter=True,
        repetition_penalty=1.2,
        condition_on_previous_text=False,
        initial_prompt=WHISPER_PROMPT,
        temperature=0.0
    )
    user_text = "".join([segment.text for segment in segments]).strip()
    t_stt = round(time.time() - t_whisper_start, 2)
    
    print(f"[Latency]: Mic captured in {t_mic}s | Whisper STT: {t_stt}s")
    return user_text

# -------------------------------------------------------------
# 2. APP LAUNCHER & WORKSPACE AUTOMATION
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
    """Launches local Windows applications instantly via command or shell start."""
    clean_app = app_name.lower().strip()
    
    # 1. Check known aliases and executables
    target_cmd = APP_SHORTCUTS.get(clean_app, clean_app)
    
    try:
        # Launch using Windows start shell
        subprocess.Popen(f"start {target_cmd}", shell=True)
        return f"Opening {app_name}, Aimz."
    except Exception as e:
        return f"Unable to launch {app_name}: {str(e)}"

def lock_workstation():
    """Locks the Windows session."""
    ctypes.windll.user32.LockWorkStation()
    return "Workstation locked, Aimz."

def minimize_all_windows():
    """Simulates Windows + D to toggle/minimize all windows."""
    VK_LWIN = 0x5B
    VK_D = 0x44
    KEYEVENTF_KEYUP = 0x0002
    
    ctypes.windll.user32.keybd_event(VK_LWIN, 0, 0, 0)
    ctypes.windll.user32.keybd_event(VK_D, 0, 0, 0)
    ctypes.windll.user32.keybd_event(VK_D, 0, KEYEVENTF_KEYUP, 0)
    ctypes.windll.user32.keybd_event(VK_LWIN, 0, KEYEVENTF_KEYUP, 0)
    return "Showing desktop, Aimz."

# -------------------------------------------------------------
# 3. HARDWARE, MARKET, WEB & MEDIA TOOLS
# -------------------------------------------------------------
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
    return f"Media command executed, Aimz."

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
    """Instantly handles apps, workstation control, hardware, media, crypto, and forex (<0.05s)."""
    clean = text.lower()
    
    # 1. Volume commands
    if "volume" in clean or "sound" in clean:
        nums = re.findall(r'\b\d+\b', clean)
        if nums:
            return set_volume_level(int(nums[0]))
        if "mute" in clean:
            return set_volume_level(0)
        if "max" in clean or "full" in clean:
            return set_volume_level(100)

    # 2. Brightness commands
    if "brightness" in clean or "dim" in clean:
        nums = re.findall(r'\b\d+\b', clean)
        if nums:
            return set_brightness(int(nums[0]))
        if "max" in clean or "full" in clean:
            return set_brightness(100)
        if "dim" in clean:
            return set_brightness(30)

    # 3. Windows & Workstation Actions
    if any(k in clean for k in ["lock pc", "lock workstation", "lock computer", "lock screen"]):
        return lock_workstation()
        
    if any(k in clean for k in ["show desktop", "minimize all", "minimize windows", "clear screen"]):
        return minimize_all_windows()

    # 4. Direct YouTube Music Instant Play
    if any(k in clean for k in ["youtube music", "yt music"]) or (clean.startswith("play") and any(w in clean for w in ["song", "track", "music"])):
        q = re.sub(r'^(play|search for|search|open)\s+(song|track|music)?\s*', '', clean)
        q = re.sub(r'\s+(on|in)?\s*(youtube music|yt music)$', '', q).strip()
        return play_youtube_music(q)

    # 5. Direct YouTube Video Instant Play
    if "youtube" in clean or "yt" in clean or clean.startswith("play video"):
        q = re.sub(r'^(play|search for|search|look up|open)\s+(video)?\s*', '', clean)
        q = re.sub(r'\s+(on|in)?\s*(youtube|yt)$', '', q).strip()
        return play_youtube_video(q)

    # 6. Media Control (Toggles existing tabs in Opera GX)
    if any(k in clean for k in ["pause", "resume", "unpause"]):
        return media_control("play_pause")
    if any(w in clean for w in ["next track", "next song", "skip track", "skip song"]):
        return media_control("next")
    if any(w in clean for w in ["previous track", "previous song", "back song"]):
        return media_control("previous")

    # 7. Hardware & System Telemetry
    if "battery" in clean or "power" in clean or "charge" in clean:
        return get_battery_status()
        
    if "spec" in clean or "ram" in clean or "cpu" in clean or "system" in clean:
        return get_system_specs()
        
    if "browser" in clean or "chrome" in clean or "google" in clean or "open internet" in clean:
        return open_browser()

    # 8. App Launching triggers (e.g. "open vs code", "launch spotify", "start task manager")
    if any(clean.startswith(prefix) for prefix in ["open ", "launch ", "start "]):
        app_target = re.sub(r'^(open|launch|start)\s+(app|application)?\s*', '', clean).strip()
        if app_target:
            return launch_application(app_target)

    # 9. Fast Crypto Spot Prices
    if "bitcoin" in clean or "btc" in clean:
        return get_crypto_price("bitcoin")
    if "ethereum" in clean or "eth" in clean:
        return get_crypto_price("ethereum")
    if "solana" in clean or "sol" in clean:
        return get_crypto_price("solana")
    if "ripple" in clean or "xrp" in clean:
        return get_crypto_price("ripple")

    # 10. Fast Currency & Forex Lookups
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

    # 11. Fast Weather Lookup
    if "weather" in clean:
        match = re.search(r'weather\s+(?:in|for)?\s*([a-zA-Z\s]+)', clean)
        loc = match.group(1).strip() if match else ""
        weather_res = get_live_weather(loc)
        if weather_res:
            return weather_res

    return None

# -------------------------------------------------------------
# 4. MAIN EVENT LOOP
# -------------------------------------------------------------
def main():
    speak("E.V. online and operational.")
    
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

    while True:
        try:
            user_input = listen(silence_limit=0.6, threshold=450)
            
            if not user_input or len(user_input.strip()) < 2:
                continue
                
            print(f"[You]: {user_input}")
            
            clean_input = user_input.lower().replace(".", "").strip()
            if any(word in clean_input for word in ["shutdown", "shut down", "goodbye", "exit", "stop"]):
                speak("Systems going offline. Goodbye, Aimz.")
                break

            # 1. Fast direct actions (hardware, apps, windows, media, financial APIs) (<0.05s)
            fast_reply = handle_direct_commands(clean_input)
            if fast_reply:
                print(f"[Fast-Path Executed]: Direct response.")
                messages.append({"role": "user", "content": user_input})
                messages.append({"role": "assistant", "content": fast_reply})
                speak(fast_reply)
                continue

            # 2. Live web search path
            needs_search = any(trigger in clean_input for trigger in SEARCH_TRIGGERS)
            search_context = ""
            
            if needs_search:
                print("[E.V.]: Searching the web for real-time data...")
                query = re.sub(r'^(search for|search|look up|what is|what\'s|whats|who is|who\'s|whos|tell me about)\s+', '', clean_input).strip()
                search_results = web_search(query)
                search_context = f"\n\nLIVE SEARCH RESULTS:\n{search_results}"

            # 3. Conversational generation
            user_message_content = user_input + search_context
            messages.append({"role": "user", "content": user_message_content})
            
            if len(messages) > 5:
                messages = [system_prompt] + messages[-3:]

            t_llm_start = time.time()
            response = ollama.chat(model='llama3.2:3b', messages=messages)
            reply = response.message.content
            print(f"[Latency]: LLM generation: {round(time.time() - t_llm_start, 2)}s")
            
            messages.append({"role": "assistant", "content": reply})
            speak(reply)

        except KeyboardInterrupt:
            speak("Interrupted. Systems going offline.")
            break
        except Exception as e:
            print(f"\n[Error]: {e}")
            speak("I encountered a brief processing error.")

if __name__ == "__main__":
    main()