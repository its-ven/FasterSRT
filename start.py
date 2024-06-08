import subprocess
import sys
import os

def install_and_import(package):
    try:
        __import__(package)
    except ImportError:
        print("Installing dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print("Dependencies installed.")
    finally:
        globals()[package] = __import__(package)

if __name__ == "__main__":
  install_and_import('faster_whisper')
  import autosrt
  autosrt.start()