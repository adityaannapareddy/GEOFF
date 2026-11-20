import asyncio
import base64
import json
import os
import re
import sqlite3
import uuid
from contextlib import asynccontextmanager, closing
from pathlib import Path
from urllib.parse import quote_plus

import httpx
from cryptography.fernet import Fernet, InvalidToken
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DB = DATA / "settings.sqlite3"
KEY = DATA / "secret.key"


def connect_db():
    DATA.mkdir(mode=0o700, parents=True, exist_ok=True)
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with closing(connect_db()) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS runs ("
            "id TEXT PRIMARY KEY, status TEXT, stage TEXT, keywords TEXT, expanded TEXT, "
            "results TEXT, error TEXT, updated TEXT DEFAULT CURRENT_TIMESTAMP)"
        )
        conn.execute("CREATE TABLE IF NOT EXISTS seen_posts (item_id TEXT PRIMARY KEY, first_seen_run TEXT NOT NULL)")
        conn.execute("CREATE TABLE IF NOT EXISTS geocode_cache (candidate TEXT PRIMARY KEY, country TEXT NOT NULL)")
        conn.commit()
    try:
        os.chmod(DB, 0o600)
    except OSError:
        pass


def read_settings():
    with closing(connect_db()) as conn:
        return {r["key"]: r["value"] for r in conn.execute("SELECT key,value FROM settings")}


def write_setting(key, value):
    with closing(connect_db()) as conn:
        conn.execute(
            "INSERT INTO settings(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        conn.commit()


def update_run(run_id, **fields):
    if not fields:
        return
    assignments = ", ".join(f"{key}=?" for key in fields)
    values = list(fields.values()) + [run_id]
    with closing(connect_db()) as conn:
        conn.execute(f"UPDATE runs SET {assignments}, updated=CURRENT_TIMESTAMP WHERE id=?", values)
        conn.commit()


def read_run(run_id):
    with closing(connect_db()) as conn:
        row = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Run not found.")
    result = dict(row)
    for key in ("keywords", "expanded", "results"):
        try:
            result[key] = json.loads(result[key]) if result[key] else []
        except ValueError:
            result[key] = []
    return result


class ModelSettings(BaseModel):
    base_url: str = Field(min_length=8, max_length=500)
    model: str = Field(min_length=1, max_length=200)
    api_key: str | None = Field(default=None, max_length=2000)


class BlueskySettings(BaseModel):
    handle: str = Field(min_length=3, max_length=200)
    app_password: str | None = Field(default=None, max_length=300)


class RunRequest(BaseModel):
    keywords: str = Field(min_length=1, max_length=1200)
    platform: str = Field(default="bluesky", min_length=1, max_length=20)
    seed_limit: int = Field(default=3, ge=1, le=10)
    final_limit: int = Field(default=10, ge=1, le=30)


@asynccontextmanager
async def lifespan(app):
    DATA.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not KEY.exists():
        fd = os.open(KEY, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(Fernet.generate_key())
    app.state.cipher = Fernet(KEY.read_bytes())
    init_db()
    app.state.tasks = set()
    app.state.live_page = None
    app.state.geocode_lock = asyncio.Lock()
    app.state.last_geocode = 0.0
    yield


app = FastAPI(title="GEOFF", lifespan=lifespan)


def model_config():
    s = read_settings()
    if not s.get("base_url") or not s.get("model"):
        raise HTTPException(status_code=409, detail="Set up the model in Settings first.")
    try:
        token = app.state.cipher.decrypt(s.get("api_key", "").encode()).decode() if s.get("api_key") else ""
    except InvalidToken:
        raise HTTPException(status_code=500, detail="Could not read the saved model token.")
    if "deepseek" not in s["model"].lower():
        raise HTTPException(status_code=409, detail="Set the model name to DeepSeek-V4.1-Flash in Settings.")
    return s["base_url"], s["model"], token


def chat_url(base):
    base = base.strip().rstrip("/")
    if not base.startswith(("https://", "http://")):
        raise HTTPException(status_code=422, detail="The model API address must start with http:// or https://.")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


async def ask_model(messages, timeout=120):
    base, model, token = model_config()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                chat_url(base),
                headers=headers,
                json={"model": model, "messages": messages, "temperature": 0, "stream": False},
            )
    except httpx.RequestError as exc:
        raise RuntimeError(f"Could not reach ARC: {exc}") from exc
    if response.is_error:
        body = response.text[:1000]
        if token:
            body = body.replace(token, "[redacted]")
        raise RuntimeError(f"ARC returned HTTP {response.status_code}: {body}")
    try:
        content = response.json()["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = " ".join(x.get("text", "") for x in content if isinstance(x, dict))
        return str(content).strip()
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("ARC returned a response this app could not read.") from exc


def parse_json(text):
    try:
        return json.loads(text)
    except ValueError:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except ValueError:
                pass
    return {}


def get_bsky_credentials():
    s = read_settings()
    handle = s.get("bsky_handle", "")
    encrypted = s.get("bsky_password", "")
    if not handle or not encrypted:
        raise RuntimeError("Open the Bluesky tab and save your handle and app password.")
    try:
        password = app.state.cipher.decrypt(encrypted.encode()).decode()
    except InvalidToken as exc:
        raise RuntimeError("Could not read the saved Bluesky app password.") from exc
    return handle, password



async def login_bluesky(page, handle, password):
    # Stay on the feed while checking for an existing signed-in session.
    await page.goto("https://bsky.app", wait_until="domcontentloaded", timeout=45000)
    await page.wait_for_timeout(2500)
    composer = page.locator('button[aria-label="Add image"]').first
    if await composer.count() and await composer.is_visible():
        return

    dialog = page.locator('div[role="dialog"][aria-modal="true"]').first
    if await dialog.count() and await dialog.is_visible():
        if not await dialog.locator("input").count():
            await page.keyboard.press("Escape")
            await page.wait_for_timeout(500)

    username_selectors = (
        'input[autocomplete="username"], '
        'input[placeholder*="email" i], '
        'input[placeholder*="handle" i], '
        'input[aria-label*="email" i], '
        'input[aria-label*="handle" i]'
    )
    username_box = page.locator(username_selectors).first
    password_box = page.locator(
        'input[type="password"], input[autocomplete="current-password"]'
    ).first

    async def visible(locator):
        return await locator.count() > 0 and await locator.first.is_visible()

    if not await visible(username_box) and not await visible(password_box):
        sign_in = page.locator('button[aria-label="Sign in"]').first
        if not await sign_in.count():
            sign_in = page.get_by_role(
                "button", name=re.compile("sign in|log in", re.I)
            ).first
        if not await sign_in.count():
            sign_in = page.get_by_role(
                "link", name=re.compile("sign in|log in", re.I)
            ).first

        if await sign_in.count() and await sign_in.is_visible():
            # Bluesky can show a modal overlay that blocks the page Sign in button.
            dialog = page.locator('[role="dialog"][aria-modal="true"]').last
            if await dialog.count() and await dialog.is_visible():
                close_button = dialog.locator(
                    '[aria-label*="close" i], [aria-label*="dismiss" i], '
                    'button[title*="close" i]'
                ).first
                if await close_button.count() and await close_button.is_visible():
                    await close_button.click(timeout=3000)
                else:
                    await page.keyboard.press("Escape")
                await page.wait_for_timeout(700)

            remaining_dialog = page.locator('[role="dialog"][aria-modal="true"]').last
            if await remaining_dialog.count() and await remaining_dialog.is_visible():
                details = await page.locator('[role="dialog"]').evaluate_all(
                    "(els) => els.map(e => ({text:(e.innerText||'').trim().slice(0,400), "
                    "buttons:[...e.querySelectorAll('button')].map(b => "
                    "(b.innerText||b.getAttribute('aria-label')||'').trim()).slice(0,12)}))"
                )
                raise RuntimeError(
                    f"Bluesky has a dialog covering the Sign in button. Dialog details: {details}"
                )

            await sign_in.click(timeout=5000)

            # Wait for the sign-in dialog or form to appear.
            for _ in range(20):
                if await visible(username_box) or await visible(password_box):
                    break
                await page.wait_for_timeout(500)

            # Wait for the sign-in dialog or form to appear.
            for _ in range(20):
                if await visible(username_box) or await visible(password_box):
                    break
                await page.wait_for_timeout(500)

    if not await visible(username_box) and not await visible(password_box):
        try:
            body = (await page.locator("body").inner_text())[:700]
        except Exception:
            body = "(page text unavailable)"
        try:
            buttons = await page.locator("button").evaluate_all(
                "(els) => els.map(e => ({text:(e.innerText||'').trim().slice(0,80), "
                "aria:e.getAttribute('aria-label'), title:e.getAttribute('title')})).slice(0,40)"
            )
            dialogs = await page.locator('[role="dialog"]').evaluate_all(
                "(els) => els.map(e => (e.innerText||'').trim().slice(0,250))"
            )
            inputs = await page.locator("input").evaluate_all(
                "(els) => els.map(e => ({type:e.type, placeholder:e.placeholder, "
                "aria:e.getAttribute('aria-label')}))"
            )
        except Exception as diagnostic_error:
            buttons = dialogs = inputs = f"Could not inspect page: {diagnostic_error}"

        raise RuntimeError(
            f"Bluesky opened but its sign-in form did not appear. "
            f"Page: {page.url}. Buttons: {buttons}. Dialogs: {dialogs}. "
            f"Inputs: {inputs}. Page says: {body}"
        )

    if await visible(username_box):
        await username_box.fill(handle)

    if not await visible(password_box):
        next_button = page.get_by_role(
            "button", name=re.compile("next|continue", re.I)
        ).first
        if await next_button.count() and await next_button.is_visible():
            await next_button.click()
        elif await visible(username_box):
            await username_box.press("Enter")

        for _ in range(30):
            if await visible(password_box):
                break
            await page.wait_for_timeout(500)

    if not await visible(password_box):
        try:
            body = (await page.locator("body").inner_text())[:700]
        except Exception:
            body = "(page text unavailable)"
        raise RuntimeError(
            f"Bluesky did not show the password field. "
            f"Page: {page.url}. Page says: {body}"
        )

    await password_box.fill(password)
    submit = page.locator('button[type="submit"]').first
    if await submit.count() and await submit.is_visible():
        await submit.click()
    else:
        button = page.get_by_role(
            "button", name=re.compile("sign in|log in|next|continue", re.I)
        ).last
        if not await button.count():
            raise RuntimeError("Bluesky showed the password field but no submit button.")
        await button.click()

    await page.wait_for_timeout(4000)
    if await visible(password_box):
        try:
            body = (await page.locator("body").inner_text())[:700]
        except Exception:
            body = "(page text unavailable)"
        raise RuntimeError(
            f"Bluesky kept the sign-in form open after submission. "
            f"Check the saved handle and app password. Page says: {body}"
        )

async def open_bsky():
    try:
        from playwright.async_api import async_playwright
        from playwright_stealth import Stealth
    except ImportError as exc:
        raise RuntimeError(
            "Playwright is not installed yet. In the app folder run: "
            ".venv/bin/pip install playwright playwright-stealth && .venv/bin/playwright install chromium"
        ) from exc

    handle, password = get_bsky_credentials()
    stealth_context = Stealth().use_async(async_playwright())
    manager = stealth_context
    p = await manager.__aenter__()
    try:
        profile_dir = DATA / "bluesky-profile"
        profile_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        context = await p.chromium.launch_persistent_context(
            user_data_dir=str(profile_dir),
            headless=True,
            viewport={"width": 1280, "height": 900},
        )
        page = context.pages[0] if context.pages else await context.new_page()
        app.state.live_page = page
        await login_bluesky(page, handle, password)
        return manager, context, page
    except Exception:
        await manager.__aexit__(None, None, None)
        raise


async def scrape_term(page, term, limit):
    await page.goto("https://bsky.app/search?q=" + quote_plus(term), wait_until="domcontentloaded", timeout=30000)
    try:
        await page.wait_for_selector('div[data-testid="postText"]', timeout=15000)
    except Exception:
        return []

    for _ in range(2):
        if await page.locator('div[data-testid="postText"]').count() >= limit:
            break
        await page.mouse.wheel(0, 700)
        await asyncio.sleep(1)

    raw = await page.locator('div[data-testid="postText"]').evaluate_all("""els => els.map(el => {
      let n=el;
      for(let i=0;i<8 && n;i++,n=n.parentElement) {
        const post=n.querySelector('a[href*="/post/"]');
        const profile=n.querySelector('a[href^="/profile/"]');
        if(post && profile) {
          const t=n.querySelector('time');
          const imgs=[...n.querySelectorAll('img')].map(x=>({url:x.src,alt:x.alt||''}))
            .filter(x=>x.url && !x.url.includes('avatar'));
          return {
            text:el.innerText.trim(),
            handle:(profile.getAttribute('href')||'').replace('/profile/','').split('/')[0],
            url:new URL(post.getAttribute('href'),location.origin).href,
            created_at:t ? (t.getAttribute('datetime')||t.innerText) : '',
            images:imgs.slice(0,3)
          };
        }
      }
      return {text:el.innerText.trim(),handle:'',url:'',created_at:'',images:[]};
    })""")

    out, seen = [], set()
    for item in raw:
        url = item.get("url", "")
        if not item.get("text") or not url or url in seen:
            continue
        seen.add(url)
        item["source_keyword"] = term
        out.append(item)
        if len(out) >= limit:
            break
    return out


async def add_profile_info(page, posts):
    cache = {}
    for post in posts:
        handle = post.get("handle", "")
        if not handle:
            post["bio"] = ""
            post["account_created"] = None
            post["account_age"] = "Unknown"
            continue
        if handle not in cache:
            try:
                await page.goto("https://bsky.app/profile/" + handle, wait_until="domcontentloaded", timeout=20000)
                await asyncio.sleep(0.5)
                bio = ""
                for selector in ('div[data-testid="profileHeaderDescription"]', 'div[dir="auto"]'):
                    el = page.locator(selector).first
                    if await el.count():
                        bio = (await el.inner_text()).strip()
                        if bio:
                            break
                body = (await page.locator("body").inner_text())[:5000]
                joined = re.search(r"Joined\s+([A-Za-z]+\s+\d{4})", body, re.I)
                created = joined.group(1) if joined else None
                cache[handle] = {"bio": bio[:1000], "account_created": created}
            except Exception:
                cache[handle] = {"bio": "", "account_created": None}
        post.update(cache[handle])
        post["account_age"] = "Unknown"
        if post.get("account_created"):
            try:
                from datetime import datetime
                created = datetime.strptime(post["account_created"], "%B %Y")
                years = (datetime.utcnow() - created).days // 365
                post["account_age"] = f"About {max(0, years)} years"
            except ValueError:
                pass
    return posts



async def collect_posts(terms, per_term, total_limit):
    manager, context, page = await open_bsky()
    try:
        found, by_url = [], {}
        for term in terms:
            if len(found) >= total_limit:
                break
            items = await scrape_term(page, term, per_term)
            for item in items:
                url = item.get("url", "")
                if not url:
                    continue
                if url in by_url:
                    queries = by_url[url].setdefault("queries", [])
                    if term not in queries:
                        queries.append(term)
                    continue
                item["queries"] = [term]
                by_url[url] = item
                found.append(item)
                if len(found) >= total_limit:
                    break
            await asyncio.sleep(1)
        return await add_profile_info(page, found)
    finally:
        await context.close()
        await manager.__aexit__(None, None, None)

async def image_inputs(post):
    parts = []
    async with httpx.AsyncClient(timeout=20) as client:
        for image in post.get("images", [])[:2]:
            url = image.get("url", "")
            if not url.startswith("https://"):
                continue
            try:
                response = await client.get(url)
                if response.is_error or len(response.content) > 3_000_000:
                    continue
                mime = response.headers.get("content-type", "image/jpeg").split(";")[0]
                encoded = base64.b64encode(response.content).decode()
                parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}})
            except httpx.RequestError:
                continue
    return parts


async def expand_keywords(seed_posts, seeds):
    corpus = "\n".join(
        f"Keyword={p.get('source_keyword')} | @{p.get('handle')} | Bio={p.get('bio','')} | Post={p.get('text','')}"
        for p in seed_posts[:20]
    )
    prompt = (
        "Starting keywords: " + ", ".join(seeds) + "\n\n"
        "Bluesky posts collected for those keywords:\n" + corpus + "\n\n"
        "Suggest 5 to 10 specific contextual search terms, phrases, names, or technical concepts "
        "that would find related posts. Return only JSON: "
        '{"contextual_keywords":["term", "..."]}'
    )
    text = await ask_model([
        {"role": "system", "content": "You expand social-media research search terms. Return valid JSON only."},
        {"role": "user", "content": prompt},
    ])
    data = parse_json(text)
    return [str(x).strip() for x in data.get("contextual_keywords", []) if str(x).strip()][:10]



async def analyze_post(post):
    prompt = (
        "Analyze one public Bluesky post for a research workflow. Return valid JSON only. "
        "Identify the post language using a short language code such as en, ar, fa, es, or und. "
        "If the post is not English, translate it into English; otherwise translated_text must be an empty string. "
        "Analyze intent and topics. Estimate the author's general location at the most specific supported level: "
        "city, region/state/province, and country. Explicit self-disclosure is not required. "
        "Use the supplied bio, post text, image alt text, and contextual evidence. "
        "One clear, credible location statement may suffice; otherwise weigh consistent clues together. "
        "Only use evidence actually present in the supplied material; do not assume access to other posts, "
        "images, account history, or external information. "
        "Distinguish residence or a usual base from travel, event attendance, employment, quotations, "
        "reposts, and places merely mentioned. A place mention must meaningfully connect the author to it. "
        "Language alone, names, and stereotypes must not establish location. "
        "Choose a city when evidence reasonably favors it over alternatives; do not force a city result. "
        "If cities remain similarly plausible, return only their supported common region or country. "
        "Use city, region, country formatting when supported, omitting unavailable components. "
        "Reflect uncertainty in location_confidence. In location_basis, distinguish explicit statements "
        "from contextual inference. In location_evidence, quote the supplied supporting text and briefly "
        "explain the connection without inventing facts. "
        "Never return anything more specific than a city, including a neighborhood, street address, "
        "postal code, building, coordinates, or current or live whereabouts. "
        "If there is no clear evidence, set author_location to null, confidence to none, and evidence to an empty string. "
        "Return keys: language, translated_text, intent, intent_evidence, topics, relevance, "
        "author_location, location_confidence, location_basis, location_evidence, image_summary.\\n\\n"
        f"Handle: @{post.get('handle','')}\\n"
        f"Public bio: {post.get('bio','')}\\n"
        f"Account age: {post.get('account_age','Unknown')}\\n"
        f"Post date: {post.get('created_at','Unknown')}\\n"
        f"Post text: {post.get('text','')}\\n"
        "Image alt text: " + "; ".join(i.get("alt", "") for i in post.get("images", []))
    )
    content = [{"type": "text", "text": prompt}]
    content.extend(await image_inputs(post))
    raw = await ask_model([
        {"role": "system", "content": "Analyze the supplied public post and images. Return valid JSON only."},
        {"role": "user", "content": content},
    ])
    return parse_json(raw) or {"model_response": raw}

async def normalize_author_location(candidate):
    """Resolve an inferred place to a country; do not store or return coordinates."""
    if not isinstance(candidate, str):
        return ""
    candidate = candidate.strip()[:120]
    if not candidate:
        return ""

    with closing(connect_db()) as conn:
        cached = conn.execute(
            "SELECT country FROM geocode_cache WHERE candidate=?",
            (candidate.casefold(),),
        ).fetchone()
    if cached:
        return cached["country"]

    try:
        async with app.state.geocode_lock:
            elapsed = asyncio.get_running_loop().time() - app.state.last_geocode
            if elapsed < 1.1:
                await asyncio.sleep(1.1 - elapsed)
            async with httpx.AsyncClient(
                timeout=12,
                headers={"User-Agent": "GEOFF/1.0 (local research application)"},
            ) as client:
                response = await client.get(
                    "https://nominatim.openstreetmap.org/search",
                    params={
                        "q": candidate,
                        "format": "jsonv2",
                        "addressdetails": 1,
                        "limit": 1,
                    },
                )
            app.state.last_geocode = asyncio.get_running_loop().time()

        response.raise_for_status()
        results = response.json()
        address = results[0].get("address", {}) if results else {}
        country = str(address.get("country", "")).strip()
        if country:
            with closing(connect_db()) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO geocode_cache(candidate,country) VALUES(?,?)",
                    (candidate.casefold(), country),
                )
                conn.commit()
            return country
    except Exception:
        pass

    return candidate


def output_records(run):
    records = []
    _, model, _ = model_config()

    for post in run.get("results", []):
        analysis = post.get("analysis", {})
        if not isinstance(analysis, dict):
            analysis = {}

        translated = str(analysis.get("translated_text") or "").strip()
        queries = post.get("queries") or (
            [post["source_keyword"]] if post.get("source_keyword") else []
        )

        row = {
            "item_id": post.get("url", ""),
            "platform": "bluesky",
            "author": post.get("handle", ""),
            "published_at": post.get("created_at", ""),
            "url": post.get("url", ""),
            "language": str(analysis.get("language") or "und").strip().lower(),
            "original_text": post.get("text", ""),
            "translated_text": translated,
            "translation_provider": "vt_arc" if translated else "",
            "translation_model": model if translated else "",
            "queries": "; ".join(queries),
            "geography": post.get("geography", ""),
            "run_id": run.get("id", ""),
            "is_new": "TRUE" if post.get("is_new") else "FALSE",
        }

        details = {
            "intent": analysis.get("intent", ""),
            "intent_evidence": analysis.get("intent_evidence", ""),
            "author_bio": post.get("bio", ""),
            "account_created": post.get("account_created", ""),
            "account_age": post.get("account_age", "Unknown"),
            "location_confidence": analysis.get("location_confidence", ""),
            "location_basis": analysis.get("location_basis", ""),
            "location_evidence": analysis.get("location_evidence", ""),
            "topics": analysis.get("topics", []),
            "image_summary": analysis.get("image_summary", ""),
            "analysis_error": post.get("analysis_error", ""),
        }
        records.append({"row": row, "details": details})

    return records


async def finalize_post(post, run_id):
    analysis = post.get("analysis", {})
    if not isinstance(analysis, dict):
        analysis = {}

    # Geocode only the broad place inferred by the model from profile/post evidence.
    post["geography"] = await normalize_author_location(analysis.get("author_location"))
    post["run_id"] = run_id

    with closing(connect_db()) as conn:
        cursor = conn.execute(
            "INSERT OR IGNORE INTO seen_posts(item_id,first_seen_run) VALUES(?,?)",
            (post.get("url", ""), run_id),
        )
        conn.commit()
        post["is_new"] = cursor.rowcount == 1



async def run_pipeline(run_id, seeds, seed_limit, final_limit):
    try:
        update_run(run_id, status="running", stage="Searching initial keywords")
        seed_posts = await collect_posts(
            seeds, seed_limit, max(seed_limit, seed_limit * len(seeds))
        )

        update_run(run_id, stage="Finding contextual keywords")
        expanded = await expand_keywords(seed_posts, seeds)

        update_run(
            run_id,
            expanded=json.dumps(expanded),
            stage="Searching original and contextual keywords",
        )
        all_terms = list(dict.fromkeys(seeds + expanded))
        final_posts = await collect_posts(all_terms, 4, final_limit)

        update_run(run_id, stage=f"Analyzing {len(final_posts)} posts sequentially")
        results = []
        for index, post in enumerate(final_posts, 1):
            update_run(run_id, stage=f"Analyzing post {index} of {len(final_posts)}")
            try:
                post["analysis"] = await analyze_post(post)
                await finalize_post(post, run_id)
            except Exception as exc:
                post["analysis_error"] = str(exc)[:500]
                post["run_id"] = run_id
                post["is_new"] = False
            results.append(post)

        update_run(
            run_id,
            status="complete",
            stage="Complete",
            results=json.dumps(results, ensure_ascii=False),
        )
    except Exception as exc:
        update_run(run_id, status="error", stage="Stopped", error=str(exc)[:1500])

PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>GEOFF</title>
<style>
:root{color-scheme:light;--bg:#f3f5f7;--panel:#fff;--line:#dce3e9;--text:#172535;--muted:#657487;--accent:#167c72;--accent-dark:#105f58;--soft:#e8f4f1}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1200px;margin:auto;padding:28px}
header{display:flex;justify-content:space-between;align-items:center;gap:24px;border-bottom:1px solid var(--line);padding:8px 0 20px}
h1{font-size:54px;line-height:1;letter-spacing:3px;margin:0;color:#14283b}
h2{font-size:21px;margin:4px 0 8px}
.muted,.hint{color:var(--muted);line-height:1.55}
.muted{font-size:14px;max-width:760px}
nav{display:flex;gap:8px;flex-wrap:wrap}
.tab,button{cursor:pointer;border:1px solid var(--line);border-radius:9px;padding:10px 15px;background:#fff;color:var(--text);font:600 14px system-ui,sans-serif}
.tab:hover,button:hover{border-color:var(--accent);color:var(--accent-dark)}
.tab.active,button.primary{background:var(--accent);color:#fff;border-color:var(--accent)}
button:disabled{cursor:not-allowed;opacity:.55}
.page{display:none;padding-top:26px}
.page.active{display:block}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.research-top{grid-template-columns:minmax(0,2fr) minmax(220px,1fr);align-items:start}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:20px;margin:16px 0;box-shadow:0 5px 20px rgba(28,48,70,.045)}
label{display:block;margin:12px 0 7px;font-weight:650}
input,textarea,select{width:100%;padding:12px;border:1px solid #cbd5df;border-radius:9px;background:#fff;color:var(--text);font:14px system-ui,sans-serif}
input:focus,textarea:focus,select:focus{outline:3px solid rgba(22,124,114,.16);border-color:var(--accent)}
textarea{min-height:112px;resize:vertical}
button{margin-top:13px}
.hint{font-size:12px}
.message{margin-top:10px;color:#8a4b08;white-space:pre-wrap;overflow-wrap:anywhere}
.ok{color:var(--accent-dark)}
.table-wrap{overflow:auto;max-height:70vh;border:1px solid var(--line);border-radius:9px;background:#fff}
table{border-collapse:collapse;min-width:1500px;width:100%;font-size:12px}
th,td{border-bottom:1px solid var(--line);padding:10px;text-align:left;vertical-align:top;max-width:300px;overflow-wrap:anywhere}
th{position:sticky;top:0;background:#edf2f5;color:#26394b}
td details{min-width:180px}
td pre{min-width:260px;max-height:240px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7f9;border:1px solid var(--line);padding:12px;border-radius:8px;font:12px/1.5 ui-monospace,monospace}

.live-card{overflow:hidden}
.live-heading{display:flex;justify-content:space-between;align-items:center;gap:12px}
.live-heading h2{margin:0}
.live-badge{border:1px solid #cbd5df;border-radius:999px;padding:5px 10px;color:#526477;font-size:12px;white-space:nowrap}
.live-screen{position:relative;aspect-ratio:16/9;max-height:680px;display:grid;place-items:center;overflow:hidden;border-radius:10px;background:#101722}
#live-image{display:none;width:100%;height:100%;object-fit:contain}
#live-empty{position:absolute;color:#d1d9e2;padding:18px;text-align:center}

.run-status{margin-top:14px;padding:16px 18px;border:1px solid var(--line);border-radius:12px;background:#f8fafb}
.run-status-top{display:flex;align-items:center;gap:12px}
.run-status-indicator{width:28px;height:28px;flex:0 0 28px;display:grid;place-items:center;border-radius:50%;background:#e8f4f1;color:var(--accent-dark)}
.run-spinner{display:none;width:17px;height:17px;border:2px solid #bddbd5;border-top-color:var(--accent);border-radius:50%;animation:run-spin .8s linear infinite}
.run-symbol{display:grid;place-items:center;font-weight:800}
.run-status-copy{min-width:0;flex:1}
.run-status-copy strong{display:block;font-size:14px;color:var(--text)}
#run-status-detail{margin-top:2px;color:var(--muted);font-size:12px;line-height:1.45;overflow-wrap:anywhere}
.run-percent{font-size:12px;font-weight:700;color:var(--muted);font-variant-numeric:tabular-nums}
.run-progress-track{height:6px;margin-top:13px;overflow:hidden;border-radius:99px;background:#e3e9ed}
.run-progress-bar{position:relative;width:0;height:100%;border-radius:99px;background:var(--accent);transition:width .5s ease}
.run-status[data-state="queued"] .run-progress-bar,.run-status[data-state="running"] .run-progress-bar{background:linear-gradient(90deg,var(--accent),#54b9a5,var(--accent));background-size:200% 100%;animation:run-shimmer 1.8s linear infinite}
.run-status[data-state="queued"] .run-spinner,.run-status[data-state="running"] .run-spinner{display:block}
.run-status[data-state="queued"] .run-symbol,.run-status[data-state="running"] .run-symbol{display:none}
.run-status[data-state="error"] .run-status-indicator{background:#fff0ed;color:#a33d2b}
.run-status[data-state="error"] .run-symbol{color:#a33d2b}
@keyframes run-spin{to{transform:rotate(360deg)}}
@keyframes run-shimmer{to{background-position:-200% 0}}

.account-subtabs{display:flex;gap:8px;flex-wrap:wrap;margin:18px 0}
.account-subtab{border:1px solid var(--line);border-radius:9px;padding:9px 14px;background:#fff;color:var(--text);font:600 14px system-ui,sans-serif;cursor:pointer}
.account-subtab:hover{border-color:var(--accent);color:var(--accent-dark)}
.account-subtab.active{background:var(--accent);border-color:var(--accent);color:#fff}
.account-panel{display:none}
.account-panel.active{display:block}

main{max-width:1440px;padding:18px 24px}
#research.page{padding-top:12px}
#research>h2{margin:0}
#research>p{margin:3px 0 12px}
.research-dashboard{display:grid;grid-template-columns:minmax(390px,.92fr) minmax(500px,1.08fr);gap:14px;height:calc(100vh - 240px);min-height:460px;max-height:760px}
.research-left{display:grid;grid-template-rows:auto minmax(0,1fr);gap:12px;min-height:0}
.research-controls{margin:0;padding:13px}
.research-controls .research-top{grid-template-columns:minmax(0,1.6fr) minmax(145px,1fr);gap:12px}
.research-controls .grid:not(.research-top){gap:12px}
.research-controls label{margin:6px 0 4px}
.research-controls textarea{min-height:62px;padding:9px}
.research-controls input,.research-controls select{padding:9px}
.research-controls .hint{margin:3px 0}
.research-controls button{margin-top:7px;padding:8px 12px}
.research-controls .run-status{margin-top:8px;padding:8px 11px}
.research-controls .run-progress-track{margin-top:7px}
.live-card{display:flex;flex-direction:column;min-height:0;margin:0;padding:13px}
.live-heading h2{font-size:17px}
.live-screen{flex:1;min-height:170px;max-height:none;aspect-ratio:unset}
.output-card{display:flex;flex-direction:column;min-height:0;margin:0;padding:13px}
.output-card h2{margin:0}
.output-card .table-wrap{flex:1;min-height:250px;max-height:none}
.analysis-animation{display:none;position:absolute;inset:0;align-items:center;justify-content:center;gap:clamp(16px,4vw,46px);padding:22px;color:#edf6f5;background:radial-gradient(ellipse at 30% 45%,#193b45 0%,#101a29 55%,#0b1320 100%)}
.live-screen.analysis-mode #live-image,.live-screen.analysis-mode #live-empty{display:none!important}
.live-screen.analysis-mode .analysis-animation{display:flex}
.analysis-visual{position:relative;width:clamp(130px,17vw,210px);height:clamp(130px,17vw,210px);flex:0 0 auto;display:grid;place-items:center}
.analysis-orbit{position:absolute;inset:8%;border:1px solid rgba(108,232,206,.38);border-radius:50%;animation:analysis-spin 8s linear infinite}
.analysis-orbit.orbit-two{inset:23%;border-color:rgba(132,174,255,.5);animation-direction:reverse;animation-duration:6s}
.analysis-orbit i{position:absolute;top:8%;left:17%;width:8px;height:8px;border-radius:50%;background:#6ce8ce;box-shadow:0 0 15px #6ce8ce}
.analysis-orbit.orbit-two i{top:auto;left:auto;right:5%;bottom:24%;background:#83aeff;box-shadow:0 0 14px #83aeff}
.analysis-core{width:58px;height:58px;display:grid;place-items:center;border:1px solid rgba(108,232,206,.65);border-radius:18px;background:rgba(24,66,70,.8);color:#8cf0d8;font-size:25px;font-weight:800;box-shadow:0 0 32px rgba(108,232,206,.22);animation:analysis-pulse 2.2s ease-in-out infinite}
.analysis-scan{position:absolute;left:17%;right:17%;height:1px;background:linear-gradient(90deg,transparent,#8cf0d8,transparent);box-shadow:0 0 12px #6ce8ce;animation:analysis-scan 2.6s ease-in-out infinite}
.analysis-copy{width:min(390px,48%);min-width:190px}
.analysis-kicker{display:block;margin-bottom:8px;color:#80e4cd;font-size:10px;font-weight:750;letter-spacing:1.5px}
.analysis-copy strong{display:block;color:#fff;font-size:clamp(17px,2vw,23px);line-height:1.2}
.analysis-copy p{margin:8px 0 15px;color:#b6c7d3;font-size:12px;line-height:1.5}
.analysis-meter{height:5px;overflow:hidden;border-radius:99px;background:rgba(226,240,245,.17)}
.analysis-meter span{display:block;width:8%;height:100%;border-radius:99px;background:linear-gradient(90deg,#4ec8ae,#a0ffe4);box-shadow:0 0 12px rgba(108,232,206,.65);transition:width .5s ease}
.analysis-copy small{display:block;margin-top:7px;color:#9fb2c1;font-size:10px}
.live-screen.analysis-done .analysis-orbit,.live-screen.analysis-done .analysis-core,.live-screen.analysis-done .analysis-scan{animation-play-state:paused}
@keyframes analysis-spin{to{transform:rotate(360deg)}}
@keyframes analysis-pulse{50%{transform:scale(1.08);box-shadow:0 0 44px rgba(108,232,206,.42)}}
@keyframes analysis-scan{0%,100%{top:25%;opacity:.2}50%{top:75%;opacity:1}}
@media(max-width:980px){.research-dashboard{height:auto;max-height:none;min-height:0;grid-template-columns:1fr}.research-left{grid-template-rows:auto minmax(300px,45vh)}.output-card .table-wrap{max-height:55vh}.analysis-animation{min-height:280px}}
@media(max-height:760px) and (min-width:981px){.research-dashboard{height:calc(100vh - 215px);min-height:430px}.research-controls textarea{min-height:54px}.research-controls .run-status{padding:6px 10px}}
@media(max-width:600px){.analysis-animation{flex-direction:column;gap:8px;text-align:center}.analysis-visual{width:120px;height:120px}.analysis-copy{width:100%;min-width:0}}
@media(prefers-reduced-motion:reduce){.analysis-orbit,.analysis-core,.analysis-scan{animation:none!important}}
@media(prefers-reduced-motion:reduce){.run-spinner,.run-progress-bar{animation:none;transition:none}}
@media(max-width:720px){main{padding:16px}header{align-items:flex-start;flex-direction:column}h1{font-size:45px}.grid,.research-top{grid-template-columns:1fr}}

/* GEOFF overview layout refinement */
.research-dashboard{grid-template-columns:minmax(340px,.88fr) minmax(520px,1.12fr);grid-template-rows:auto minmax(0,1fr);grid-template-areas:"controls live" "output live";height:calc(100vh - 240px);min-height:500px;max-height:800px}
.research-left{display:contents}
.research-controls{grid-area:controls}
.live-card{grid-area:live;min-height:0}
.output-card{grid-area:output}
.output-card .table-wrap{min-height:0}
.live-screen{min-height:0}
.analysis-animation{gap:clamp(14px,2.5vw,34px);padding:clamp(16px,2.8vw,30px)}
.analysis-visual{width:clamp(125px,15vw,190px);height:clamp(125px,15vw,190px)}
.analysis-copy{width:min(440px,53%);min-width:170px}
.analysis-meter{position:relative}
.analysis-meter span{width:34%;animation:analysis-meter-drift 1.8s ease-in-out infinite}
.live-screen.analysis-done .analysis-meter span{width:100%!important;transform:none;animation:none}
@keyframes analysis-meter-drift{0%{transform:translateX(-110%)}100%{transform:translateX(300%)}}

.results-empty{min-height:100%;display:grid;place-content:center;justify-items:center;gap:10px;padding:28px 24px;text-align:center;background:radial-gradient(ellipse at 50% 0%,#f5faf9 0%,#fff 65%)}
.results-empty-mark{width:54px;height:54px;display:flex;align-items:flex-end;justify-content:center;gap:5px;padding:12px;border:1px solid #d9ebe7;border-radius:17px;background:linear-gradient(145deg,#f1faf7,#fff);box-shadow:0 8px 22px rgba(22,124,114,.08)}
.results-empty-mark span{width:6px;border-radius:5px;background:linear-gradient(180deg,#56bba8,#167c72)}
.results-empty-mark span:nth-child(1){height:13px;opacity:.55}
.results-empty-mark span:nth-child(2){height:25px}
.results-empty-mark span:nth-child(3){height:18px;opacity:.75}
.results-empty-copy{max-width:380px}
.results-empty-kicker{display:block;color:var(--accent);font-size:10px;font-weight:750;letter-spacing:1.5px}
.results-empty-copy strong{display:block;margin-top:4px;color:var(--text);font-size:17px}
.results-empty-copy p{margin:5px 0 0;color:var(--muted);font-size:12px;line-height:1.55}
.results-empty-tags{display:flex;flex-wrap:wrap;justify-content:center;gap:6px;margin-top:4px}
.results-empty-tags span{padding:4px 8px;border:1px solid #e1e9ed;border-radius:999px;background:#fff;color:#647487;font-size:10px}

@media(max-width:980px){
  .research-dashboard{height:auto;max-height:none;min-height:0;grid-template-columns:minmax(0,1fr);grid-template-rows:auto auto auto;grid-template-areas:"controls" "output" "live";gap:12px}
  .research-left{display:contents}
  .output-card{min-height:330px}
  .output-card .table-wrap{max-height:55vh}
  .live-card{min-height:440px}
  .live-screen{min-height:300px}
  .analysis-animation{min-height:280px}
}
@media(max-width:600px){
  .live-card{min-height:380px}
  .live-screen{min-height:250px}
  .analysis-animation{flex-direction:column;gap:8px;text-align:center;padding:16px}
  .analysis-visual{width:120px;height:120px}
  .analysis-copy{width:100%;min-width:0}
}
@media(prefers-reduced-motion:reduce){.analysis-meter span{animation:none!important}}

</style></head><body><main>
<header><h1>GEOFF</h1><nav><button class="tab active" data-page="research">Research</button><button class="tab" data-page="accounts">Accounts</button><button class="tab" data-page="settings">Settings</button></nav></header>
<section id="research" class="page active">
<h2>Research</h2><p class="muted">Choose a platform and keywords. GEOFF searches, finds contextual terms, then analyzes posts with available profile and image context.</p>
<div class="research-dashboard"><div class="research-left"><div class="card research-controls"><div class="grid research-top"><div><label for="keywords">Keywords</label><textarea id="keywords" placeholder="One keyword or phrase per line"></textarea></div><div><label for="platform">Social media</label><select id="platform"><option value="bluesky">Bluesky</option><option value="reddit" disabled>Reddit — coming soon</option><option value="x" disabled>X — coming soon</option></select><p class="hint">Bluesky is available now. Other platforms will be added later.</p></div></div>
<div class="grid"><div><label for="seed-limit">Posts per starting keyword</label><input id="seed-limit" type="number" min="1" max="10" value="3"></div>
<div><label for="final-limit">Maximum posts to analyze</label><input id="final-limit" type="number" min="1" max="30" value="10"></div></div>
<button class="primary" id="start-run">Start research run</button><div id="run-status" class="run-status" data-state="idle" role="status" aria-live="polite"><div class="run-status-top"><div class="run-status-indicator"><span id="run-status-spinner" class="run-spinner"></span><span id="run-status-symbol" class="run-symbol">•</span></div><div class="run-status-copy"><strong id="run-status-title">Ready to research</strong><div id="run-status-detail">Choose a platform and enter keywords to begin.</div></div><span id="run-status-percent" class="run-percent">0%</span></div><div class="run-progress-track"><div id="run-progress-bar" class="run-progress-bar"></div></div></div></div>
<div class="card live-card"><div class="live-heading"><div><h2>Live Playwright Activity</h2><p id="live-state" class="hint">Connecting to the browser stream…</p></div><span class="live-badge">15 FPS target</span></div><div class="live-screen" id="live-screen">
<img id="live-image" alt="Live view of Playwright in Bluesky">
<div id="live-empty">The browser view appears while GEOFF is browsing.</div>
<div id="live-analysis" class="analysis-animation" aria-live="polite">
  <div class="analysis-visual" aria-hidden="true">
    <div class="analysis-orbit orbit-one"><i></i><i></i></div>
    <div class="analysis-orbit orbit-two"><i></i><i></i></div>
    <div class="analysis-core">G</div>
    <div class="analysis-scan"></div>
  </div>
  <div class="analysis-copy">
    <span class="analysis-kicker">VT ARC \u00b7 DEEPSEEK</span>
    <strong id="analysis-title">Analyzing collected posts</strong>
    <p id="analysis-detail">Reviewing post intent, author context, and available images.</p>
    <div class="analysis-meter"><span id="analysis-meter-fill"></span></div>
    <small id="analysis-meter-label">Preparing analysis</small>
  </div>
</div>
</div></div></div><div class="card output-card"><h2>Research results</h2><button id="download-csv" disabled>Download CSV</button><div id="output" class="table-wrap"><div class="results-empty" role="status"><div class="results-empty-mark" aria-hidden="true"><span></span><span></span><span></span></div><div class="results-empty-copy"><span class="results-empty-kicker">RESEARCH OUTPUT</span><strong>Your results will appear here</strong><p>Start a research run to see analyzed posts, author context, and location findings.</p><div class="results-empty-tags"><span>Post intent</span><span>Author bio</span><span>Location</span><span>Images</span></div></div></div></div></div></div></section><section id="accounts" class="page"><h2>Accounts</h2><p class="muted">Manage the social accounts GEOFF can use for research.</p><div class="account-subtabs"><button type="button" class="account-subtab active" data-account="account-bluesky">Bluesky</button><button type="button" class="account-subtab" data-account="account-reddit">Reddit</button><button type="button" class="account-subtab" data-account="account-x">X</button></div><div id="account-bluesky" class="account-panel active"><h3>Bluesky</h3><p class="muted">Use a Bluesky app password, not your main account password. The app encrypts it on this server and keeps a local browser session.</p>
<div class="card"><label for="handle">Bluesky handle</label><input id="handle" placeholder="malikcollier02.bsky.social" value="malikcollier02.bsky.social">
<label for="bsky-password">Bluesky app password</label><input id="bsky-password" type="password" autocomplete="new-password" placeholder="Leave blank to keep the saved app password">
<button id="save-bsky">Save account</button><button class="primary" id="test-bsky">Test Bluesky login</button><div id="bsky-message" class="message"></div></div>
</div><div id="account-reddit" class="account-panel"><div class="card"><h3>Reddit</h3><p class="muted">Reddit account setup will appear here when Reddit research is added.</p><p class="hint">This section does not save or use Reddit credentials yet.</p></div></div><div id="account-x" class="account-panel"><div class="card"><h3>X</h3><p class="muted">X account setup will appear here when X research is added.</p><p class="hint">This section does not save or use X credentials yet.</p></div></div></section>
<section id="settings" class="page"><h2>Model settings</h2><p class="muted">Use VT ARC and DeepSeek-V4.1-Flash for text and image analysis. Model credentials remain encrypted on this server.</p>
<div class="card"><label for="url">API address</label><input id="url" value="https://llm-api.arc.vt.edu/api/v1/">
<label for="model">Model name</label><input id="model" value="DeepSeek-V4.1-Flash">
<label for="key">API token</label><input id="key" type="password" placeholder="Leave blank to keep the saved token">
<button id="save-model">Save model settings</button><button class="primary" id="test-model">Test model</button><div id="model-message" class="message"></div></div>
</section>
<script>
const $=id=>document.getElementById(id);
async function api(path,method="GET",body){const o={method,headers:{"Content-Type":"application/json"}};if(body)o.body=JSON.stringify(body);const r=await fetch(path,o);const raw=await r.text();let d;try{d=JSON.parse(raw)}catch(e){throw Error("Server returned HTTP "+r.status+": "+raw.slice(0,500))}if(!r.ok)throw Error(d.detail||"Request failed");return d}
function msg(id,text,ok=false){
  if(id==="run-status"){
    const box=$("run-status"),raw=String(text||"");
    const parts=raw.split(" \u00b7 "),known=["queued","running","complete","error"];
    let state=known.includes(parts[0])?parts[0]:"idle";
    let stage=known.includes(parts[0])?parts.slice(1).join(" \u00b7 "):raw;
    let title="Ready to research",detail=stage||"Choose a platform and enter keywords to begin.",progress=0;
    if(state==="queued"||/starting|connecting/i.test(raw)){
      state="queued";title="Preparing your research";
      detail="Connecting to Bluesky and preparing the keyword search.";progress=4;
    }else if(state==="complete"){
      title="Research complete";detail="Your results are ready.";progress=100;
    }else if(state==="error"||/could not|failed|error|enter at least/i.test(raw)){
      state="error";title="Could not start research";detail=stage||raw;progress=0;
    }else if(state==="running"){
      title="Research in progress";detail=stage||"Working through the research steps.";progress=18;
      if(stage.includes("initial keywords")){
        title="Searching starting keywords";detail="Collecting relevant public Bluesky posts.";progress=14;
      }else if(stage.includes("contextual keywords")){
        title="Finding related search terms";detail="DeepSeek is finding useful context in the first posts.";progress=30;
      }else if(stage.includes("original and contextual")){
        title="Searching expanded keywords";detail="Searching with your original and contextual keywords.";progress=43;
      }else{
        const post=stage.match(/Analyzing post (\\d+) of (\\d+)/i);
        const batch=stage.match(/Analyzing (\\d+) posts sequentially/i);
        if(post){
          const current=Number(post[1]),total=Math.max(1,Number(post[2]));
          title="Analyzing post "+current+" of "+total;
          detail="Reviewing intent, language, author bio, location evidence, and available images.";
          progress=52+43*current/total;
        }else if(batch){
          const total=Number(batch[1]);
          title="Preparing post analysis";
          detail=total?"DeepSeek will analyze "+total+" posts one at a time.":"No matching posts were found.";
          progress=52;
        }
      }
    }
    box.dataset.state=state;
    $("run-status-title").textContent=title;
    $("run-status-detail").textContent=detail;
    $("run-progress-bar").style.width=Math.max(0,Math.min(100,progress))+"%";
    $("run-status-percent").textContent=Math.round(progress)+"%";
    const busy=state==="queued"||state==="running";
    $("run-status-spinner").style.display=busy?"block":"none";
    $("run-status-symbol").style.display=busy?"none":"grid";
    $("run-status-symbol").textContent=state==="complete"?"\u2713":state==="error"?"!":"\u2022";
    return;
  }
  $(id).textContent=text;
  $(id).className="message"+(ok?" ok":"");
}
document.querySelectorAll(".tab").forEach(b=>b.onclick=()=>{document.querySelectorAll(".tab,.page").forEach(x=>x.classList.remove("active"));b.classList.add("active");$(b.dataset.page).classList.add("active")});document.querySelectorAll(".account-subtab").forEach(b=>b.onclick=()=>{document.querySelectorAll(".account-subtab,.account-panel").forEach(x=>x.classList.remove("active"));b.classList.add("active");$(b.dataset.account).classList.add("active")});
async function load(){try{let m=await api("/api/model");$("url").value=m.base_url||$("url").value;$("model").value=m.model||$("model").value;if(m.has_key)$("key").placeholder="Saved token; leave blank to keep it";let b=await api("/api/bluesky/account");$("handle").value=b.handle||"malikcollier02.bsky.social";if(b.has_password)$("bsky-password").placeholder="Saved app password; leave blank to keep it"}catch(e){}}
$("save-model").onclick=async()=>{try{await api("/api/model","PUT",{base_url:$("url").value,model:$("model").value,api_key:$("key").value||null});$("key").value="";msg("model-message","Model settings saved.",true)}catch(e){msg("model-message",e.message)}};
$("test-model").onclick=async()=>{msg("model-message","Sending test\u2026");try{let r=await api("/api/model/test","POST",{});msg("model-message","Connected: "+r.reply,true)}catch(e){msg("model-message",e.message)}};
$("save-bsky").onclick=async()=>{try{await api("/api/bluesky/account","PUT",{handle:$("handle").value,app_password:$("bsky-password").value||null});$("bsky-password").value="";msg("bsky-message","Bluesky account saved.",true)}catch(e){msg("bsky-message",e.message)}};
$("test-bsky").onclick=async()=>{msg("bsky-message","Opening Bluesky with Playwright\u2026");try{let r=await api("/api/bluesky/test","POST",{});msg("bsky-message",r.message,!!r.ok)}catch(e){msg("bsky-message",e.message)}};
$("start-run").onclick=async()=>{let keywords=$("keywords").value.trim();if(!keywords){msg("run-status","Enter at least one keyword.");return}try{let r=await api("/api/runs","POST",{keywords,platform:$("platform").value,seed_limit:Number($("seed-limit").value),final_limit:Number($("final-limit").value)});poll(r.id)}catch(e){msg("run-status",e.message)}};
function esc(v){return String(v??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]))}
const fields=["item_id","platform","author","published_at","url","language","original_text","translated_text","translation_provider","translation_model","queries","geography","run_id","is_new"];
function showEmptyResults(title,detail){
  $("output").innerHTML='<div class="results-empty" role="status"><div class="results-empty-mark" aria-hidden="true"><span></span><span></span><span></span></div><div class="results-empty-copy"><span class="results-empty-kicker">RESEARCH OUTPUT</span><strong>'+esc(title)+'</strong><p>'+esc(detail)+'</p><div class="results-empty-tags"><span>Post intent</span><span>Author bio</span><span>Location</span><span>Images</span></div></div></div>';
}
function showRows(records){
  if(!records.length){showEmptyResults("No posts found","Try broadening your keywords or increasing the post limits, then run the search again.");return}
  let h="<table><thead><tr>"+fields.map(x=>"<th>"+x+"</th>").join("")+"<th>Analysis details</th></tr></thead><tbody>";
  for(const x of records){
    const r=x.row,d=x.details||{};
    h+="<tr>"+fields.map(k=>"<td>"+esc(r[k])+"</td>").join("")+
      "<td><details><summary>Intent, bio, location, images</summary><pre>"+
      esc(JSON.stringify(d,null,2))+"</pre></details></td></tr>";
  }
  $("output").innerHTML=h+"</tbody></table>";
}


let liveRunActive=false;
let liveCurrentStage="";
let liveHasFrame=false;
let liveLastFrameAt=0;
let liveSameSince=0;
let liveFrameSignature="";
let liveViewMode="idle";
let liveFrameHashing=false;

function setLiveStatus(mode,message){
  if(liveViewMode===mode && $("live-state").textContent===message)return;
  liveViewMode=mode;
  $("live-state").textContent=message;
}
function resetLiveView(){
  liveHasFrame=false;
  liveLastFrameAt=0;
  liveSameSince=0;
  liveFrameSignature="";
  liveViewMode="";
  const screen=$("live-screen");
  screen.classList.remove("analysis-mode","analysis-done");
  $("live-analysis").style.display="none";
  $("live-image").style.display="none";
  $("live-image").removeAttribute("src");
  delete $("live-image").dataset.frameUrl;
  $("live-empty").style.display="block";
  $("live-state").textContent="Waiting for Playwright activity\u2026";
}
function beginLiveRun(){
  resetLiveView();
  liveRunActive=true;
  liveCurrentStage="Queued";
}
function renderAnalysisStage(stage){
  const current=String(stage||"");
  const kicker=document.querySelector(".analysis-kicker");
  let title="Preparing research";
  let detail="Getting the next research step ready.";
  let label="Preparing";
  let progress=5;

  const post=current.match(/Analyzing post (\d+) of (\d+)/i);
  const queue=current.match(/Analyzing (\d+) posts sequentially/i);

  if(post){
    const index=Number(post[1]);
    const total=Math.max(1,Number(post[2]));
    kicker.textContent="WORKFLOW STEP 4 OF 4 \u00b7 POST ANALYSIS";
    title="Analyzing post "+index+" of "+total;
    detail="Reviewing post intent, author bio, location evidence, and available images.";
    label="Post "+index+" of "+total+" \u00b7 workflow step 4 of 4";
    progress=55+45*Math.min(index,total)/total;
  }else if(queue){
    const total=Number(queue[1]);
    kicker.textContent="WORKFLOW STEP 4 OF 4 \u00b7 POST ANALYSIS";
    title="Preparing post-by-post analysis";
    detail="DeepSeek will analyze "+total+" collected posts one at a time.";
    label="Analysis queue \u00b7 "+total+" posts";
    progress=55;
  }else if(/^Searching initial keywords/i.test(current)){
    kicker.textContent="WORKFLOW STEP 1 OF 4";
    title="Searching starting keywords";
    detail="Finding public posts that match your original keywords.";
    label="Step 1 of 4 \u00b7 Initial search";
    progress=12;
  }else if(/^Finding contextual keywords/i.test(current)){
    kicker.textContent="WORKFLOW STEP 2 OF 4";
    title="Finding contextual keywords";
    detail="Using the first posts to identify related terms for a deeper search.";
    label="Step 2 of 4 \u00b7 Context discovery";
    progress=32;
  }else if(/^Searching original and contextual keywords/i.test(current)){
    kicker.textContent="WORKFLOW STEP 3 OF 4";
    title="Searching expanded keywords";
    detail="Searching again with your original keywords and the related terms.";
    label="Step 3 of 4 \u00b7 Expanded search";
    progress=52;
  }else if(current){
    kicker.textContent="RESEARCH WORKFLOW";
    title=current;
    detail="GEOFF is working through this research step.";
    label="Research in progress";
  }else{
    kicker.textContent="RESEARCH WORKFLOW";
  }

  $("analysis-title").textContent=title;
  $("analysis-detail").textContent=detail;
  $("analysis-meter-label").textContent=label;
  $("analysis-meter-fill").style.width=Math.round(progress)+"%";
}

function updateLivePhase(run){
  liveCurrentStage=run.stage||"";
  liveRunActive=run.status==="queued"||run.status==="running";
  const screen=$("live-screen");

  if(screen.classList.contains("analysis-mode") && liveRunActive){
    renderAnalysisStage(liveCurrentStage);
    const analyzing=/^Analyzing/i.test(liveCurrentStage);
    setLiveStatus(
      analyzing?"paused-analysis":"paused",
      "Playwright paused \u00b7 "+(liveCurrentStage||"research in progress")
    );
  }

  if((run.status==="complete"||run.status==="error")&&screen.classList.contains("analysis-mode")){
    screen.classList.add("analysis-done");
    document.querySelector(".analysis-kicker").textContent=
      run.status==="complete"?"RESEARCH COMPLETE":"RESEARCH STOPPED";
    $("analysis-title").textContent=run.status==="complete"?"Analysis complete":"Run stopped";
    $("analysis-detail").textContent=run.status==="complete"
      ?"Your collected posts are ready to review."
      :"See the research status for details.";
    $("analysis-meter-fill").style.width=run.status==="complete"?"100%":"0%";
    $("analysis-meter-label").textContent=run.status==="complete"?"Complete":"Stopped";
    setLiveStatus("finished",run.status==="complete"?"Analysis complete":"Research run stopped");
  }
}
async function liveFrameHash(blob){
  const bytes=new Uint8Array(await blob.arrayBuffer());
  let hash=2166136261;
  for(let i=0;i<bytes.length;i++)hash=Math.imul(hash^bytes[i],16777619);
  return bytes.length+":"+hash;
}
function showLiveFrame(url,signature){
  const screen=$("live-screen"),image=$("live-image");
  const previous=image.dataset.frameUrl;
  screen.classList.remove("analysis-mode","analysis-done");
  $("live-analysis").style.display="none";
  $("live-empty").style.display="none";
  image.src=url;
  image.dataset.frameUrl=url;
  image.style.display="block";
  if(previous)setTimeout(()=>URL.revokeObjectURL(previous),1500);
  liveFrameSignature=signature;
  liveSameSince=Date.now();
  setLiveStatus("active","Playwright active \u00b7 browsing Bluesky");
}
function showPausedAnimation(){
  const screen=$("live-screen");
  if(screen.classList.contains("analysis-mode"))return;

  screen.classList.add("analysis-mode");
  $("live-image").style.display="none";
  $("live-empty").style.display="none";
  $("live-analysis").style.display="flex";

  renderAnalysisStage(liveCurrentStage);
  const analyzing=/^Analyzing/i.test(liveCurrentStage);
  setLiveStatus(
    analyzing?"paused-analysis":"paused",
    "Playwright paused \u00b7 "+(liveCurrentStage||"view unchanged for 8 seconds")
  );
}
function checkLiveStall(){
  if(!liveHasFrame||$("live-screen").classList.contains("analysis-mode"))return;
  const now=Date.now();
  const unchangedFor=now-liveSameSince;
  const noFramesFor=now-liveLastFrameAt;
  if(unchangedFor>=8000||noFramesFor>=8000)showPausedAnimation();
}
async function poll(id){
  msg("run-status","Starting\u2026");
  beginLiveRun();
  updateLivePhase({status:"queued",stage:"Queued"});
  const timer=setInterval(async()=>{
    try{
      let r=await api("/api/runs/"+id);
      msg("run-status",r.status+" \u00b7 "+r.stage);
      updateLivePhase(r);
      if(r.status==="complete"){
        $("download-csv").disabled=false;
        $("download-csv").onclick=()=>window.location="/api/runs/"+id+"/export.csv";
        let o=await api("/api/runs/"+id+"/output");
        showRows(o.records||[]);
      }
      if(r.status==="error")$("output").textContent=r.error||"Run failed.";
      if(r.status==="complete"||r.status==="error")clearInterval(timer);
    }catch(e){
      clearInterval(timer);
      msg("run-status",e.message);
    }
  },2000);
}

function connectLive(){
  const scheme=location.protocol==="https:"?"wss":"ws";
  const socket=new WebSocket(scheme+"://"+location.host+"/ws/live");
  socket.binaryType="blob";
  socket.onopen=()=>{
    if(!liveHasFrame)setLiveStatus("waiting","Waiting for Playwright activity\u2026");
  };
  socket.onmessage=async event=>{
    if(typeof event.data==="string"){
      if($("live-screen").classList.contains("analysis-mode"))return;
      if(!liveHasFrame){
        try{
          const state=JSON.parse(event.data);
          setLiveStatus("waiting",liveRunActive
            ?"Waiting for Playwright to open Bluesky\u2026"
            :(state.message||"Waiting for a research run"));
        }catch(e){}
      }
      return;
    }

    liveLastFrameAt=Date.now();
    if(liveFrameHashing)return;
    liveFrameHashing=true;
    try{
      const signature=await liveFrameHash(event.data);
      liveHasFrame=true;
      if(!liveSameSince)liveSameSince=Date.now();
      if(signature!==liveFrameSignature){
        const url=URL.createObjectURL(event.data);
        showLiveFrame(url,signature);
      }
    }catch(e){
      setLiveStatus("stream-error","Receiving the Playwright preview");
    }finally{
      liveFrameHashing=false;
    }
  };
  socket.onclose=()=>{
    if(!liveHasFrame)setLiveStatus("waiting","Live preview disconnected \u00b7 reconnecting\u2026");
    setTimeout(connectLive,2000);
  };
  socket.onerror=()=>{
    if(!liveHasFrame)setLiveStatus("waiting","Connecting to the Playwright preview\u2026");
  };
}
setInterval(checkLiveStall,500);
connectLive();

load();</script></main></body></html>"""


@app.websocket("/ws/live")
async def live_browser(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            page = getattr(app.state, "live_page", None)
            if page is None or page.is_closed():
                await websocket.send_json({
                    "state": "idle",
                    "message": "Waiting for GEOFF to open the Playwright browser.",
                })
                await asyncio.sleep(0.6)
                continue

            started = asyncio.get_running_loop().time()
            try:
                frame = await page.screenshot(
                    type="jpeg",
                    quality=55,
                    scale="css",
                )
                await websocket.send_bytes(frame)
            except Exception:
                await websocket.send_json({
                    "state": "idle",
                    "message": "Waiting for the browser page.",
                })
                await asyncio.sleep(0.4)
                continue

            elapsed = asyncio.get_running_loop().time() - started
            await asyncio.sleep(max(0, (1 / 15) - elapsed))
    except WebSocketDisconnect:
        pass
    except Exception:
        try:
            await websocket.close()
        except Exception:
            pass


@app.get("/", response_class=HTMLResponse)
async def home():
    return PAGE


@app.get("/api/model")
async def get_model():
    s = read_settings()
    return {"base_url": s.get("base_url", ""), "model": s.get("model", ""), "has_key": bool(s.get("api_key"))}


@app.put("/api/model")
async def save_model(config: ModelSettings):
    base = config.base_url.strip().rstrip("/")
    if not base.startswith(("https://", "http://")):
        raise HTTPException(status_code=422, detail="Enter a complete HTTP or HTTPS API address.")
    write_setting("base_url", base)
    write_setting("model", config.model.strip())
    if config.api_key:
        write_setting("api_key", app.state.cipher.encrypt(config.api_key.encode()).decode())
    return {"ok": True}


@app.post("/api/model/test")
async def test_model():
    text = await ask_model([
        {"role": "system", "content": "Reply exactly: ARC connection ready."},
        {"role": "user", "content": "Confirm the connection."},
    ])
    return {"reply": text[:240]}


@app.get("/api/bluesky/account")
async def get_bluesky_account():
    s = read_settings()
    return {"handle": s.get("bsky_handle", ""), "has_password": bool(s.get("bsky_password"))}


@app.put("/api/bluesky/account")
async def save_bluesky_account(config: BlueskySettings):
    write_setting("bsky_handle", config.handle.strip())
    if config.app_password:
        write_setting("bsky_password", app.state.cipher.encrypt(config.app_password.encode()).decode())
    return {"ok": True}


@app.post("/api/bluesky/test")
async def test_bluesky():
    manager = context = None
    try:
        manager, context, page = await open_bsky()
        await page.goto("https://bsky.app/", wait_until="domcontentloaded", timeout=30000)
        return {"ok": True, "message": "Bluesky login is ready."}
    except Exception as exc:
        # Return HTTP 200 so the browser proxy does not replace the useful
        # diagnostic with its generic 502 HTML page.
        return {
            "ok": False,
            "message": f"Bluesky test failed: {type(exc).__name__}: {str(exc)[:1200]}",
        }
    finally:
        if context:
            try:
                await context.close()
            except Exception:
                pass
        if manager:
            try:
                await manager.__aexit__(None, None, None)
            except Exception:
                pass

@app.post("/api/runs")
async def start_run(request: RunRequest):
    if request.platform != "bluesky":
        raise HTTPException(status_code=422, detail="Only Bluesky research is available right now.")
    _, model, _ = model_config()
    if "deepseek" not in model.lower():
        raise HTTPException(status_code=409, detail="Set model to DeepSeek-V4.1-Flash in Settings.")
    if not read_settings().get("bsky_handle"):
        raise HTTPException(status_code=409, detail="Save a Bluesky account in the Bluesky tab first.")
    seeds = list(dict.fromkeys(x.strip() for x in re.split(r"[\n,;]+", request.keywords) if x.strip()))[:10]
    run_id = uuid.uuid4().hex[:12]
    with closing(connect_db()) as conn:
        conn.execute(
            "INSERT INTO runs(id,status,stage,keywords,expanded,results) VALUES(?,?,?,?,?,?)",
            (run_id, "queued", "Queued", json.dumps(seeds), "[]", "[]"),
        )
        conn.commit()
    task = asyncio.create_task(run_pipeline(run_id, seeds, request.seed_limit, request.final_limit))
    app.state.tasks.add(task)
    task.add_done_callback(app.state.tasks.discard)
    return {"id": run_id}


@app.get("/api/runs/{run_id}/output")
async def get_run_output(run_id: str):
    run = read_run(run_id)
    return {"records": output_records(run)}


@app.get("/api/runs/{run_id}/export.csv")
async def export_run_csv(run_id: str):
    import csv
    import io

    run = read_run(run_id)
    fields = [
        "item_id", "platform", "author", "published_at", "url", "language",
        "original_text", "translated_text", "translation_provider", "translation_model",
        "queries", "geography", "run_id", "is_new",
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for record in output_records(run):
        writer.writerow(record["row"])

    filename = f"geoff-{run_id}.csv"
    return Response(
        content=buffer.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str):
    return read_run(run_id)
