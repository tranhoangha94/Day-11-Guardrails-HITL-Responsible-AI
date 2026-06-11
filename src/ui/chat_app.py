"""
Deprecated — use Streamlit instead:
    streamlit run ui/streamlit_app.py
"""
import subprocess
import sys
from pathlib import Path


def main():
    app = Path(__file__).resolve().parent / "streamlit_app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app)], check=True)


if __name__ == "__main__":
    main()
