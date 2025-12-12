import sys
from pathlib import Path

# Ensure repo root is on sys.path for shared/frontend imports when run on Hugging Face Spaces
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:
    pass

from frontend.app import main

if __name__ == "__main__":
    main()
