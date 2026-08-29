import urllib.request
import os

piper_dir = os.path.abspath("piper")

# Direct download links with the correct repository subfolder path
onnx_url = "https://huggingface.co/jgkawell/jarvis/resolve/main/en/en_GB/jarvis/medium/jarvis-medium.onnx"
json_url = "https://huggingface.co/jgkawell/jarvis/resolve/main/en/en_GB/jarvis/medium/jarvis-medium.onnx.json"

onnx_path = os.path.join(piper_dir, "jarvis-medium.onnx")
json_path = os.path.join(piper_dir, "jarvis-medium.onnx.json")

# User-Agent header to prevent HuggingFace request blocks
headers = {'User-Agent': 'Mozilla/5.0'}

def download(url, dest):
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp, open(dest, 'wb') as f:
        f.write(resp.read())

print("Downloading custom J.A.R.V.I.S. ONNX voice model (~60MB)... Please wait.")
download(onnx_url, onnx_path)

print("Downloading J.A.R.V.I.S. JSON configuration...")
download(json_url, json_path)

print("\n[Success]: J.A.R.V.I.S. neural voice models downloaded into your 'piper' folder!")