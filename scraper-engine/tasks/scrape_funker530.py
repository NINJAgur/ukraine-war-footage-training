"""
scraper-engine/tasks/scrape_funker530.py

Celery task: scrape latest video posts from Funker530.
Uses Playwright (headless Chromium) + BeautifulSoup.

Flow:
  1. Navigate funker530.com category pages
  2. Extract video post URLs + metadata
  3. For each new URL (not yet in DB), create a Clip record and dispatch download
"""
import asyncio
import hashlib
import logging
import re
from datetime import datetime
from typing import Optional
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, Browser, Page
from sqlalchemy.dialects.postgresql import insert as pg_insert

from celery_app import celery_app
from config import settings
from db.models import Clip, ClipSource, ClipStatus
from db.session import get_session

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────
FUNKER530_CATEGORIES = [
    "/videos/",
    "/ukraine/",
]
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


# ── Helpers ───────────────────────────────────────────────────────────

def canonical_url(url: str) -> str:
    """Normalize a URL for consistent hashing (strip query, fragments, www.)."""
    parsed = urlparse(url)
    host = parsed.netloc.lstrip("www.")
    path = parsed.path.rstrip("/")
    return f"{parsed.scheme}://{host}{path}"


def url_hash(url: str) -> str:
    return hashlib.sha256(canonical_url(url).encode()).hexdigest()


def extract_post_metadata(soup: BeautifulSoup, post_url: str) -> dict:
    """Extract title, description, and published date from a post page."""
    title = ""
    description = ""
    published_at: Optional[datetime] = None

    # Title
    h1 = soup.find("h1")
    if h1:
        title = h1.get_text(strip=True)

    # Meta description
    meta_desc = soup.find("meta", attrs={"name": "description"})
    if meta_desc:
        description = meta_desc.get("content", "")

    # Published date (common patterns)
    time_tag = soup.find("time")
    if time_tag and time_tag.get("datetime"):
        try:
            published_at = datetime.fromisoformat(
                time_tag["datetime"].replace("Z", "+00:00")
            )
        except ValueError:
            pass

    return {
        "title": title[:500] if title else None,
        "description": description[:2000] if description else None,
        "published_at": published_at,
    }


async def _find_video_urls_on_page(page: Page, category_url: str, max_pages: int) -> list[dict]:
    """
    Paginate through a Funker530 category page and collect video post links.
    Returns a list of dicts: {url, title, description, published_at}.
    """
    posts: list[dict] = []
    current_url = category_url

    for page_num in range(1, max_pages + 1):
        logger.info(f"Funker530: scraping page {page_num} — {current_url}")

        try:
            await page.goto(current_url, wait_until="domcontentloaded", timeout=30_000)
            await page.wait_for_timeout(int(settings.SCRAPE_DELAY_SECONDS * 1000))
        except Exception as exc:
            logger.warning(f"Failed to load {current_url}: {exc}")
            break

        html = await page.content()
        soup = BeautifulSoup(html, "lxml")

        # Find article/post links — Funker530 uses <article> or card-style divs
        post_links: list[str] = []
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            full_url = urljoin(settings.FUNKER530_BASE_URL, href)
            # Filter to article/video post URLs (avoid category pages, tags, etc.)
            parsed = urlparse(full_url)
            if (
                parsed.netloc
                and "funker530.com" in parsed.netloc
                and len(parsed.path.strip("/").split("/")) >= 1
                and not any(skip in parsed.path for skip in ["/category/", "/tag/", "/page/", "/author/"])
                and parsed.path not in ["/", "/videos/", "/ukraine/"]
            ):
                post_links.append(full_url)

        # De-duplicate within this page
        post_links = list(dict.fromkeys(post_links))

        for post_url in post_links[:20]:  # max 20 posts per page
            try:
                await page.goto(post_url, wait_until="domcontentloaded", timeout=20_000)
                post_html = await page.content()
                post_soup = BeautifulSoup(post_html, "lxml")
                metadata = extract_post_metadata(post_soup, post_url)
                posts.append({"url": post_url, **metadata})
            except Exception as exc:
                logger.warning(f"Failed to load post {post_url}: {exc}")
                continue

        # Find "next page" link
        next_link = soup.find("a", string=re.compile(r"next|›|»", re.I))
        if not next_link or not next_link.get("href"):
            break
        current_url = urljoin(settings.FUNKER530_BASE_URL, next_link["href"])

    return posts


async def _run_scrape() -> dict:
    """Main async scraping logic. Returns summary dict."""
    new_count = 0
    skipped_count = 0
    error_count = 0

    async with async_playwright() as p:
        browser: Browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=HEADERS["User-Agent"],
            viewport={"width": 1920, "height": 1080},
        )
        page = await context.new_page()

        try:
            all_posts: list[dict] = []
            for category in FUNKER530_CATEGORIES:
                category_url = settings.FUNKER530_BASE_URL + category
                posts = await _find_video_urls_on_page(
                    page, category_url, settings.SCRAPE_MAX_PAGES
                )
                all_posts.extend(posts)

            # De-duplicate across categories by URL
            seen: set[str] = set()
            unique_posts = []
            for post in all_posts:
                h = url_hash(post["url"])
                if h not in seen:
                    seen.add(h)
                    unique_posts.append({**post, "url_hash": h})

            logger.info(f"Funker530: found {len(unique_posts)} unique posts")

            # Insert new Clips; skip existing (ON CONFLICT DO NOTHING)
            with get_session() as session:
                for post in unique_posts:
                    try:
                        stmt = (
                            pg_insert(Clip)
                            .values(
                                url=post["url"],
                                url_hash=post["url_hash"],
                                source=ClipSource.FUNKER530,
                                title=post.get("title"),
                                description=post.get("description"),
                                published_at=post.get("published_at"),
                                status=ClipStatus.PENDING,
                            )
                            .on_conflict_do_nothing(index_elements=["url_hash"])
                            .returning(Clip.id)
                        )
                        result = session.execute(stmt)
                        row = result.fetchone()
                        if row:
                            new_count += 1
                            logger.debug(f"New clip: {post['url']}")
                        else:
                            skipped_count += 1
                    except Exception as exc:
                        logger.error(f"DB error for {post['url']}: {exc}")
                        error_count += 1

        finally:
            await context.close()
            await browser.close()

    return {
        "source": "funker530",
        "new": new_count,
        "skipped": skipped_count,
        "errors": error_count,
    }


# ── Celery Task ───────────────────────────────────────────────────────

@celery_app.task(
    bind=True,
    name="tasks.scrape_funker530.scrape_funker530",
    queue="default",
    autoretry_for=(Exception,),
    max_retries=3,
    default_retry_delay=120,
)
def scrape_funker530(self) -> dict:
    """
    Scrape latest video posts from Funker530.
    Uses a Redis lock to prevent overlapping Beat executions.
    """
    import redis as redis_lib

    r = redis_lib.from_url(settings.REDIS_URL)
    lock_key = "lock:scrape_funker530"
    lock_ttl = 3600  # 1 hour

    if not r.set(lock_key, self.request.id, ex=lock_ttl, nx=True):
        logger.info(f"[{self.request.id}] scrape_funker530 already running — skipping")
        return {"status": "skipped", "reason": "lock_held"}

    logger.info(f"[{self.request.id}] scrape_funker530 started")
    try:
        result = asyncio.run(_run_scrape())
        logger.info(f"[{self.request.id}] scrape_funker530 completed: {result}")
        return result
    except Exception as exc:
        logger.error(f"[{self.request.id}] scrape_funker530 failed: {exc}", exc_info=True)
        raise
    finally:
        r.delete(lock_key)
