from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA = PROJECT_ROOT / "data"
RAW = DATA / "raw"
CLEANED = DATA / "cleaned"
PROCESSED = DATA / "processed"
OUTPUTS = PROJECT_ROOT / "outputs"

for p in (DATA, RAW, CLEANED, PROCESSED, OUTPUTS):
    p.mkdir(parents=True, exist_ok=True)
