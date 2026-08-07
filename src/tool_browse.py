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

#!/usr/bin/env python3
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

# -*- coding: utf-8 -*-

import os
import time
import json
import sqlite3
import asyncio
import aiohttp
import threading
from typing import List, Dict, Optional, Tuple
from contextlib import contextmanager
from urllib.parse import urlsplit, urlunsplit, urlencode, parse_qsl

from crawl4ai import AsyncWebCrawler

# =========================
# Config via env (no changes)
# =========================
CACHE_DB_PATH = os.getenv("BROWSE_CACHE_DB", "outputs/cache/browse_cache.sqlite")
CACHE_TTL_SEC = int(os.getenv("BROWSE_CACHE_TTL_SEC", "0")) or None
JINA_API_KEY  = os.getenv("JINA_API_KEY", "")
IGNORE_JINA   = os.getenv("IGNORE_JINA", "false").lower() == "true"
JINA_PREFIX   = os.getenv("JINA_READER_URL_PREFIX", "https://r.jina.ai/")
HTTP_TIMEOUT  = int(os.getenv("BROWSE_HTTP_TIMEOUT", "12"))
BUSY_TIMEOUT  = int(os.getenv("BROWSE_SQLITE_BUSY_TIMEOUT_MS", "10000"))

# =========================
# Small helpers (no changes)
# =========================
def canonical_url(u: str) -> str:
    u = (u or "").strip()
    sp = urlsplit(u)
    netloc = sp.netloc.lower()
    q = urlencode(sorted([(k, v) for k, v in parse_qsl(sp.query, keep_blank_values=False)]))
    path = sp.path if sp.path != '/' else ''
    return urlunsplit((sp.scheme, netloc, path, q, sp.fragment))

def _ensure_dir(p: str):
    d = os.path.dirname(os.path.abspath(p))
    os.makedirs(d, exist_ok=True)

# =========================
# Process-safe SQLite cache (no changes)
# =========================
class CacheStore:
    # (Your CacheStore code remains exactly the same)
    def __init__(self, db_path: str, ttl_sec: Optional[int] = None):
        self.db_path = db_path
        self.ttl_sec = ttl_sec
        self._local = threading.local()
        _ensure_dir(db_path)
        conn = self._conn()
        with conn:
            conn.executescript("""
                PRAGMA journal_mode=WAL;
                PRAGMA synchronous=NORMAL;
                CREATE TABLE IF NOT EXISTS cache (
                  url TEXT PRIMARY KEY,
                  content BLOB NOT NULL,
                  meta TEXT,
                  fetched_at INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_cache_time ON cache(fetched_at);
            """)

    def _conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self.db_path, timeout=BUSY_TIMEOUT/1000.0, isolation_level=None)
            c.execute("PRAGMA busy_timeout = %d;" % BUSY_TIMEOUT)
            c.execute("PRAGMA journal_mode = WAL;")
            c.execute("PRAGMA synchronous = NORMAL;")
            self._local.conn = c
        return c

    @contextmanager
    def _cursor(self):
        cur = self._conn().cursor()
        try:
            yield cur
        finally:
            cur.close()

    def get(self, url: str) -> Optional[Dict]:
        key = canonical_url(url)
        now = int(time.time())
        with self._cursor() as cur:
            cur.execute("SELECT content, meta, fetched_at FROM cache WHERE url=?", (key,))
            row = cur.fetchone()
        if not row: return None
        content, meta_json, ts = row
        if self.ttl_sec is not None and now - ts > self.ttl_sec: return None
        meta = json.loads(meta_json) if meta_json else {}
        if isinstance(content, bytes):
            try: content = content.decode("utf-8", errors="replace")
            except Exception: content = content.decode("latin-1", errors="replace")
        return {"content": content, "meta": meta, "fetched_at": ts}

    def set(self, url: str, content: str, meta: Optional[Dict] = None):
        key = canonical_url(url)
        meta_json = json.dumps(meta or {}, ensure_ascii=False)
        now = int(time.time())
        with self._cursor() as cur:
            cur.execute("""
                INSERT INTO cache(url, content, meta, fetched_at) VALUES(?, ?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                  content=excluded.content, meta=excluded.meta, fetched_at=excluded.fetched_at
            """, (key, content, meta_json, now))

# =========================
# ASYNC Fetchers (no changes)
# =========================
async def jina_read_async(session: aiohttp.ClientSession, url: str) -> Optional[str]:
    if IGNORE_JINA: return None
    headers = {"Authorization": f"Bearer {JINA_API_KEY}"} if JINA_API_KEY else {}
    try:
        async with session.get(f"{JINA_PREFIX}{url}", headers=headers) as resp:
            if resp.status == 200:
                text = await resp.text()
                if text and text.strip(): return text
    except Exception: pass
    return None

async def crawl4ai_read_async(url: str) -> Optional[str]:
    try:
        async with AsyncWebCrawler() as crawler:
            res = await crawler.arun(url=url)
            if not res: return None
            md = getattr(res, "markdown", None)
            if md and md.strip(): return md
            html = getattr(res, "html", None)
            if html and str(html).strip(): return str(html)
    except Exception: pass
    return None

# =========================
# Main ASYNC Browse Tool (CORRECTED)
# =========================
class BrowseTool:
    def __init__(self, cache_db: str = CACHE_DB_PATH, ttl_sec: Optional[int] = CACHE_TTL_SEC):
        self.cache = CacheStore(cache_db, ttl_sec=ttl_sec)
        
    async def _fetch_one_url(self, session: aiohttp.ClientSession, url: str) -> Tuple[str, str]:
        text = await jina_read_async(session, url)
        source = "jina"
        if not text:
            text = await crawl4ai_read_async(url)
            source = "crawl4ai"
        return text or "", source

    async def browse(self, urls: List[str], session: aiohttp.ClientSession) -> Dict[str, Dict[str, str]]:
        out: Dict[str, Dict[str, str]] = {}
        urls_to_check = list(set(u.strip() for u in urls if u.strip()))
        urls_to_fetch = []

        # 1) Concurrently check cache for all URLs in background threads
        if urls_to_check:
            # Create a list of tasks to run the blocking cache.get() in threads
            cache_check_tasks = [asyncio.to_thread(self.cache.get, url) for url in urls_to_check]
            # Wait for all cache checks to complete
            cached_results = await asyncio.gather(*cache_check_tasks)
            
            for url, cached in zip(urls_to_check, cached_results):
                if cached and cached.get("content"):
                    out[url] = {"content": cached["content"], "source": "cache"}
                else:
                    urls_to_fetch.append(url)
        
        # 2) Concurrently fetch all URLs that were not in the cache
        if urls_to_fetch:
            fetch_tasks = [self._fetch_one_url(session, url) for url in urls_to_fetch]
            results = await asyncio.gather(*fetch_tasks)
            
            # Prepare cache writing tasks (which are also blocking)
            cache_write_tasks = []
            for url, (text, source) in zip(urls_to_fetch, results):
                if text:
                    out[url] = {"content": text, "source": source}
                    # Add the blocking cache.set() call to a list of tasks to run in threads
                    cache_write_tasks.append(asyncio.to_thread(self.cache.set, url, text, meta={"source": source}))
                else:
                    out[url] = {"content": "", "source": "error"}
            
            # 3) Concurrently write all new results to cache in the background
            if cache_write_tasks:
                await asyncio.gather(*cache_write_tasks)
        
        # Ensure the output dict has keys for all original input URLs
        final_out = {u: out.get(u.strip(), {"content": "", "source": "invalid"}) for u in urls}
        return final_out

# =========================
# Example usage (no changes)
# =========================
async def main():
    tool = BrowseTool()
    urls = [
        "https://arxiv.org/abs/2305.10601",
        "https://en.wikipedia.org/wiki/Google",
    ]
    
    timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        start_time = time.time()
        result = await tool.browse(urls, session=session)
        end_time = time.time()
        
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    for u, r in result.items():
        print(f"\n=== {u} ===")
        print(f"source: {r['source']}")
        print((r["content"] or "")[:500])

if __name__ == "__main__":
    asyncio.run(main())