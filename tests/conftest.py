import os
import sys
import tempfile
from pathlib import Path

import pytest

# Tests must never reach a real LLM configured in the developer's .env, nor touch the
# signs the user has taught.
os.environ["LLM_BASE_URL"] = ""
os.environ["SIGNS_PATH"] = str(Path(tempfile.mkdtemp(prefix="signs-test-")) / "signs.json")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402
from app.ocr.manager import OCRManager  # noqa: E402


@pytest.fixture(scope="session")
def manager():
    m = OCRManager.from_settings(Settings.from_env())
    if not m.primary.available:
        pytest.skip(f"Tesseract unavailable: {m.primary.reason}")
    yield m
    m.close()
