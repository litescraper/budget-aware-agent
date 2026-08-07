# Copyright 2026 The BATS Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import ast
import asyncio
import json
import time
import asyncio
import aiohttp
import sqlite3
import random
from typing import Any, List, Dict, Optional

# ---------- Global cache path ----------
SEARCH_DB_CACHE = "outputs/cache/search_cache.sqlite3"

# Assume load_dotenv() is called and env variables are set
from dotenv import load_dotenv
load_dotenv()

# ---------- Search backend configuration ----------
# "cse"    - Google Custom Search JSON API (used in the paper; requires CSE_API_KEYS
#            and CSE_ID; Google has closed this API to new customers and will
#            discontinue it on 2027-01-01)
# "serper" - serper.dev Google-results API (requires SERPER_API_KEY)
# Selected via the SEARCH_BACKEND env var or the agents' --search_backend flag.
def get_search_backend() -> str:
    return os.environ.get("SEARCH_BACKEND", "cse").strip().lower()

CSE_API_KEYS = [k.strip() for k in os.environ.get("CSE_API_KEYS", "").split(",") if k.strip()]
CSE_ID = os.environ.get("CSE_ID", "")
SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "")
SERPER_URL = "https://google.serper.dev/search"
RATE_LIMITED_KEYS = {}
COOLDOWN_SECONDS = 600 # 10 minutes cooldown for rate-limited keys


def check_search_credentials():
    """Fail fast with a clear message if the selected backend's keys are missing."""
    backend = get_search_backend()
    if backend == "cse":
        if not CSE_API_KEYS or not CSE_ID:
            raise ValueError(
                "SEARCH_BACKEND=cse requires CSE_API_KEYS (one or more comma-separated "
                "Google CSE API keys) and CSE_ID in your .env. Alternatively set "
                "SEARCH_BACKEND=serper with a SERPER_API_KEY (see env.example).")
    elif backend == "serper":
        if not SERPER_API_KEY:
            raise ValueError("SEARCH_BACKEND=serper requires SERPER_API_KEY in your .env.")
    else:
        raise ValueError(f"Unknown SEARCH_BACKEND: '{backend}' (expected 'cse' or 'serper').")


def get_query_list(text: str) -> Optional[List[str]]:
    pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    matches = pattern.findall(text)
    if not matches:
        return None
    last_match = matches[-1].strip()
    try:
        parsed = ast.literal_eval(last_match)
        if isinstance(parsed, list):
            return [str(q) for q in parsed]
        elif isinstance(parsed, str):
            return [parsed]
    except (ValueError, SyntaxError):
        # If literal_eval fails, it might be a single query without quotes
        if last_match:
            return [last_match]
    return None

def get_query(text, tag="tool_call"):
    pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL)
    matches = pattern.findall(text)
    queries = []

    for m in matches:
        m = m.strip()
        # 1. Try to parse as Python list
        try:
            parsed = ast.literal_eval(m)
            if isinstance(parsed, list):
                queries.extend(str(q).strip() for q in parsed)
                continue
        except Exception:
            pass
        # 2. If it's a multi-line string
        if "\n" in m:
            queries.extend(line.strip() for line in m.splitlines() if line.strip())
        else:
            queries.append(m)
    queries = [q.strip() for q in queries if q.strip() and not q.startswith("(") and not q.startswith("[") and not q.startswith("#")]
    if not queries:
        return None
    return queries


# ---------- SQLite Cache Helpers ----------

def _ensure_dir(p: str):
    """Ensure the directory for the given file path exists."""
    d = os.path.dirname(os.path.abspath(p))
    os.makedirs(d, exist_ok=True)

def _connect_db(db_path: str = SEARCH_DB_CACHE) -> sqlite3.Connection:
    """
    Create a SQLite connection configured for multi-process read/write.
    """
    _ensure_dir(db_path)
    conn = sqlite3.connect(
        db_path,
        timeout=30.0,
        isolation_level=None,
        check_same_thread=False,
    )
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA busy_timeout=30000;")
    return conn

def _init_cache_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
    CREATE TABLE IF NOT EXISTS search_cache (
        query TEXT PRIMARY KEY,
        items_json TEXT NOT NULL,
        updated_at INTEGER NOT NULL
    );
    """)

def _load_cache(conn: sqlite3.Connection, query_key: str, max_age_seconds: Optional[int] = None) -> Optional[List[Dict[str, Any]]]:
    row = conn.execute("SELECT items_json, updated_at FROM search_cache WHERE query = ?;", (query_key,)).fetchone()
    if not row:
        return None
    items_json, updated_at = row
    if max_age_seconds is not None and (int(time.time()) - int(updated_at)) > max_age_seconds:
        return None
    try:
        return json.loads(items_json)
    except Exception:
        return None

def _save_cache(conn: sqlite3.Connection, query_key: str, items: List[Dict[str, Any]]) -> None:
    payload = json.dumps(items, ensure_ascii=False)
    now_ts = int(time.time())
    conn.execute("""
    INSERT INTO search_cache(query, items_json, updated_at)
    VALUES (?, ?, ?)
    ON CONFLICT(query) DO UPDATE SET
        items_json=excluded.items_json,
        updated_at=excluded.updated_at;
    """, (query_key, payload, now_ts))

def _dedup_by_link(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen, out = set(), []
    for it in items:
        link = it.get("link") or it.get("formattedUrl") or it.get("cacheId") or json.dumps(it, sort_keys=True)
        if link in seen:
            continue
        seen.add(link)
        out.append(it)
    return out


# ---------- Format results ----------
def _format_results_from_list(items: List[Dict[str, Any]]) -> str:
    lines = []
    for idx, item in enumerate(items):
        # print(f"Processing item {idx+1}/{len(items)}: {item}")
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        url = item.get("link") or item.get("formattedUrl") or "NA"
        lines.append(f"{idx+1}. ({title}) [{url}] {snippet}")
    return "\n".join(lines)


# ---------- Search tool ----------
async def search_with_sqlite_cache_async(
    session: aiohttp.ClientSession,
    query: str,
    *,
    num_results: int = 5,
    max_retries: int = 6,
    per_page: int = 10,
    max_age_seconds: Optional[int] = None,
) -> str:
    from os import getpid
    backend = get_search_backend()
    cache_key = query.strip().lower()
    if backend != "cse":
        # keep per-backend cache entries separate (results differ across backends)
        cache_key = f"{backend}::{cache_key}"

    conn = _connect_db(SEARCH_DB_CACHE)
    _init_cache_table(conn)

    # 1) Cache hit
    cached_items = _load_cache(conn, cache_key, max_age_seconds=max_age_seconds)
    if cached_items:
        if len(cached_items) >= num_results:
            print(f"[PID:{getpid()}] ✅ SQLite FULL HIT: '{query}' ({len(cached_items)} cached)", flush=True)
            return _format_results_from_list(cached_items[:num_results])
        else:
            print(f"[PID:{getpid()}] ✅ PARTIAL HIT: '{query}' ({len(cached_items)} cached)", flush=True)
            return _format_results_from_list(cached_items[:num_results])
    else:
        print(f"[PID:{getpid()}] 🎯 MISS: '{query}' — fetching.", flush=True)
        cached_items = []

    # 2) Access API
    all_items: List[Dict[str, Any]] = list(cached_items)

    if backend == "serper":
        # serper.dev: single POST; 'organic' items share the CSE title/link/snippet keys
        for attempt in range(max_retries):
            try:
                async with session.post(
                    SERPER_URL,
                    headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
                    json={"q": query, "num": max(num_results, 10)},
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                all_items = _dedup_by_link(all_items + data.get("organic", []))
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    return f"❌ Search failed: {e}"
                print(f"[PID:{getpid()}] ⚠️ Serper attempt {attempt+1} failed: {e}. Retrying...", flush=True)
                await asyncio.sleep(min(30, 2 ** attempt))
        if not all_items:
            return "No results found."
        try:
            conn.execute("BEGIN IMMEDIATE;")
            latest = _load_cache(conn, cache_key, None) or []
            merged = _dedup_by_link(latest + all_items)
            _save_cache(conn, cache_key, merged)
            conn.execute("COMMIT;")
        except sqlite3.OperationalError:
            _save_cache(conn, cache_key, all_items)
        finally:
            conn.close()
        return _format_results_from_list(all_items[:num_results])

    pages_needed = (num_results + per_page - 1) // per_page
    start_index = 1 + (len(cached_items) // per_page) * per_page

    # --- MODIFIED: API key management ---
    now = time.time()
    # Filter out any keys that are still in the penalty box
    available_keys = [
        key for key in CSE_API_KEYS 
        if key not in RATE_LIMITED_KEYS or now > RATE_LIMITED_KEYS[key]
    ]

    if not available_keys:
        return "❌ Search failed: All API keys are currently on cooldown."

    random.shuffle(available_keys)
    api_keys = available_keys # Use the filtered & shuffled list
    current_key_index = 0

    for page in range(pages_needed):
        if len(all_items) >= num_results:
            break
        for attempt in range(max_retries):
            if current_key_index >= len(api_keys):
                return f"❌ Search failed: All {len(api_keys)} API keys were rate-limited or failed."
            try:
                params = {
                    "q": query,
                    "key": api_keys[current_key_index],
                    "cx": CSE_ID,
                    "start": start_index,
                    "num": min(per_page, num_results - len(all_items)),
                }
                params["quotaUser"] = f"user_{random.randint(1, 1000)}"
                async with session.get("https://www.googleapis.com/customsearch/v1", params=params) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                items = data.get("items", [])
                if not items:
                    break
                all_items.extend(items)
                all_items = _dedup_by_link(all_items)
                next_pages = data.get("queries", {}).get("nextPage", [])
                if next_pages:
                    start_index = next_pages[0]["startIndex"]
                else:
                    break
                break
            except aiohttp.ClientResponseError as e:
                # If it's a 429 error, switch the API key and retry immediately
                if e.status == 429:
                    failed_key = api_keys[current_key_index]
                    expiry_time = time.time() + COOLDOWN_SECONDS
                    RATE_LIMITED_KEYS[failed_key] = expiry_time

                    print(f"[PID:{getpid()}] 🧊 API key rate-limited (429). Placing on cooldown for {COOLDOWN_SECONDS / 60:.0f} mins.", flush=True)
                    current_key_index += 1
                    continue # Immediately try again with the next available key

                # For other HTTP errors, use the original backoff logic
                if attempt == max_retries - 1:
                    return f"❌ Search failed with HTTP Error: {e}"
                print(f"[PID:{getpid()}] ⚠️ HTTP error {e.status} on attempt {attempt+1}. Retrying...", flush=True)
                await asyncio.sleep(min(30, 2 ** attempt))

            except Exception as e:
                # For non-HTTP errors, also use the original backoff logic
                if attempt == max_retries - 1:
                    return f"❌ Search failed: {e}"
                print(f"[PID:{getpid()}] ⚠️ Search attempt {attempt+1} failed: {e}. Retrying...", flush=True)
                await asyncio.sleep(min(30, 2 ** attempt))

    if not all_items:
        return "No results found."

    # 3) Write back to SQLite
    try:
        conn.execute("BEGIN IMMEDIATE;")
        latest = _load_cache(conn, cache_key, None) or []
        merged = _dedup_by_link(latest + all_items)
        _save_cache(conn, cache_key, merged)
        conn.execute("COMMIT;")
    except sqlite3.OperationalError:
        _save_cache(conn, cache_key, all_items)
    finally:
        conn.close()

    return _format_results_from_list(all_items[:num_results])
