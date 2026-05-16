"""Streamlit entrypoint: yangın video analiz paneli.

Modüler uygulama kodu ``src/ui/`` altindedir. Çalıştırma::
    streamlit run src/07_ui.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ui.app import main  # noqa: E402

if __name__ == "__main__":
    main()
