# GEOFF

Source-only share copy. It keeps the DeepSeek model default and prefills the Bluesky handle `malikcollier02.bsky.social`.

It does not include an ARC API key, Bluesky app password, local database, encryption key, Playwright browser profile, or Cloudflare Tunnel configuration. Each person must enter their own ARC API key and Bluesky app password.

## Run on Ubuntu

1. `python3 -m venv .venv`
2. `.venv/bin/python -m pip install -r requirements.txt`
3. `.venv/bin/python -m playwright install chromium`
4. `.venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8000`
5. Open `http://127.0.0.1:8000`

The app creates a local `data/` directory when run. Keep it private and do not upload it.
