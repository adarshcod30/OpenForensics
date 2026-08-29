import os, sys, pathlib
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
