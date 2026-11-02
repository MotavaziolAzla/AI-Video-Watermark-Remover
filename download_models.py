import os
import shutil
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# 1. ProPainter weights
pp_weights_dir = BASE_DIR / "models" / "ProPainter" / "weights"
pp_weights_dir.mkdir(parents=True, exist_ok=True)

pp_urls = {
    "ProPainter.pth": "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth",
    "raft-things.pth": "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/raft-things.pth",
    "recurrent_flow_completion.pth": "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/recurrent_flow_completion.pth"
}

# 2. E2FGVI-HQ weights
e2_weights_dir = BASE_DIR / "models" / "E2FGVI" / "release_model"
e2_weights_dir.mkdir(parents=True, exist_ok=True)

e2_urls = {
    "E2FGVI-HQ-CVPR22.pth": "https://huggingface.co/ZhihaoLI/E2FGVI-HQ/resolve/main/E2FGVI-HQ-CVPR22.pth"
}

# 3. LaMa weights
local_checkpoints = BASE_DIR / "checkpoints"
local_checkpoints.mkdir(parents=True, exist_ok=True)
torch_hub_checkpoints = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
torch_hub_checkpoints.mkdir(parents=True, exist_ok=True)

lama_urls = {
    "big-lama.pt": "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"
}

def download_file(url, target_path):
    if os.path.exists(target_path) and os.path.getsize(target_path) > 1000000:
        print(f"[Skip] {target_path.name} already exists ({os.path.getsize(target_path)/1e6:.1f} MB)")
        return
    print(f"[Download] Fetching {target_path.name} from {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response, open(target_path, "wb") as out_file:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024 # 1MB
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                print(f"{target_path.name}: {downloaded/1e6:.1f}/{total/1e6:.1f} MB ({(downloaded/total)*100:.1f}%)", end="\r", flush=True)
    print(f"\n[OK] {target_path.name} downloaded successfully!")

if __name__ == "__main__":
    print("=== Checking & Downloading Model Weights ===")
    for fname, url in pp_urls.items():
        download_file(url, pp_weights_dir / fname)
        
    for fname, url in e2_urls.items():
        download_file(url, e2_weights_dir / fname)

    for fname, url in lama_urls.items():
        tgt = local_checkpoints / fname
        download_file(url, tgt)
        if tgt.exists() and not (torch_hub_checkpoints / fname).exists():
            try:
                shutil.copy2(tgt, torch_hub_checkpoints / fname)
            except Exception:
                pass
        
    print("\n✅ All weights verified on disk!")
