# Quickstart

The full, current instructions are in **`docs/SETUP.md`**. The short version:

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m pytest tests -q
python scripts/compare_approaches.py
python platform/server.py
```

(On macOS / Linux activate with `source venv/bin/activate`.)

* Tests: expect `126 passed, 5 skipped` (the skipped ones are optional browser tests).
* `compare_approaches.py`: expect `REGRESSION CHECK: PASS`.
* `platform/server.py`: open http://localhost:8765 (3D) or http://localhost:8765/classic.html (2D).
* No Python at all: double-click `platform/index.html` or `platform/classic.html` for recorded replays.

What everything means and why it was built this way: `docs/FULL_GUIDE.md`.
The v1 instructions this file used to contain are in `README.md` below the v2 box.
