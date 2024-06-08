import subprocess
import sys
import os

def install_and_import(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    finally:
        globals()[package] = __import__(package)

if "__name__" == "__main__":
  print("Installing dependencies...")
  install_and_import('faster_whisper')
  print("Dependencies installed.")
  import autosrt
  autosrt.start()
