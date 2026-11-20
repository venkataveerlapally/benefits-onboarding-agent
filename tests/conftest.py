import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["DISABLE_LLM"] = "1"  # tests are deterministic and free: never call the API
