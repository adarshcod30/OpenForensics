"""Legacy entry point.

Streamlit Community Cloud is configured to serve this path, and that setting
lives in the Streamlit dashboard rather than the repository — so the file has
to keep working even though the application now lives in app.py.

Executing app.py rather than importing it matters: Streamlit runs its script
top to bottom on every interaction, and an imported module would only execute
once, on first import.
"""
from pathlib import Path

_app = Path(__file__).with_name("app.py")
exec(compile(_app.read_text(), str(_app), "exec"))
