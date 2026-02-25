"""
setup.py — One-click setup script for the Self-Evolving Knowledge System.

Usage:
    python setup.py

What it does:
    1. Checks Python version
    2. Creates a virtual environment (if not already in one)
    3. Installs dependencies (CPU-only torch + requirements.txt)
    4. Initialises the SQLite database
    5. Creates required directories
    6. Checks if Ollama is reachable
"""

import os
import sys
import subprocess
import shutil


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(BASE_DIR, ".venv")


def banner(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}\n")


def run(cmd, **kwargs):
    print(f"  > {cmd}")
    result = subprocess.run(cmd, shell=True, **kwargs)
    if result.returncode != 0:
        print(f"  ❌ Command failed with code {result.returncode}")
        return False
    return True


def main():
    banner("Self-Evolving Knowledge System — Setup")

    # 1. Python version check
    print(f"Python: {sys.version}")
    if sys.version_info < (3, 9):
        print("❌ Python 3.9+ is required.")
        sys.exit(1)
    print("✅ Python version OK\n")

    # 2. Virtual environment
    in_venv = sys.prefix != sys.base_prefix
    pip_cmd = "pip"

    if not in_venv:
        if not os.path.exists(VENV_DIR):
            print("Creating virtual environment...")
            run(f'"{sys.executable}" -m venv "{VENV_DIR}"')
        
        # Determine pip path in venv
        if os.name == "nt":
            pip_cmd = os.path.join(VENV_DIR, "Scripts", "pip.exe")
            python_cmd = os.path.join(VENV_DIR, "Scripts", "python.exe")
        else:
            pip_cmd = os.path.join(VENV_DIR, "bin", "pip")
            python_cmd = os.path.join(VENV_DIR, "bin", "python")

        print(f"✅ Virtual environment at {VENV_DIR}")
        print(f"   Activate it with:")
        if os.name == "nt":
            print(f"   .venv\\Scripts\\activate")
        else:
            print(f"   source .venv/bin/activate")
        print()
    else:
        pip_cmd = "pip"
        python_cmd = sys.executable
        print("✅ Already in a virtual environment\n")

    # 3. Install dependencies
    banner("Installing Dependencies")

    # Install CPU-only torch first
    print("Installing PyTorch (CPU-only)...")
    run(f'"{pip_cmd}" install torch --index-url https://download.pytorch.org/whl/cpu')

    # Install requirements
    req_file = os.path.join(BASE_DIR, "requirements.txt")
    print(f"\nInstalling from {req_file}...")
    run(f'"{pip_cmd}" install -r "{req_file}"')

    # Install pytest for testing
    run(f'"{pip_cmd}" install pytest')

    # 4. Create directories
    banner("Creating Directories")
    dirs = [
        os.path.join(BASE_DIR, "data"),
        os.path.join(BASE_DIR, "knowledge_base", "chroma_db"),
        os.path.join(BASE_DIR, "knowledge_base", "graph"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"  ✅ {d}")

    # 5. Initialise database
    banner("Initialising Database")
    # We need to run this using the venv python
    init_script = f"""
import sys
sys.path.insert(0, r"{BASE_DIR}")
from utils.database import init_db
init_db()
print("  ✅ SQLite database initialised")
"""
    run(f'"{python_cmd}" -c "{init_script}"')

    # 6. Check Ollama
    banner("Checking Ollama")
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            print(f"  ✅ Ollama is running. Available models: {models}")
            if not any("qwen2.5" in m for m in models):
                print(f"\n  ⚠️  Model 'qwen2.5:7b' not found. Pull it with:")
                print(f"     ollama pull qwen2.5:7b")
        else:
            raise Exception("Bad status")
    except Exception:
        print("  ⚠️  Ollama is not running or not installed.")
        print("     Install from: https://ollama.com")
        print("     Then run: ollama pull qwen2.5:7b")

    # Done
    banner("Setup Complete!")
    print("Next steps:")
    print(f"  1. Activate the venv:  .venv\\Scripts\\activate  (Windows)")
    print(f"  2. Start Ollama:       ollama serve")
    print(f"  3. Pull the model:     ollama pull qwen2.5:7b")
    print(f"  4. Run the app:        streamlit run app.py")
    print(f"  5. Run tests:          pytest tests/ -v")
    print()


if __name__ == "__main__":
    main()
