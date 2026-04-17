"""
test_scrape_live.py — smoke test for real Funker530 + GeoConfirmed scraping.

Funker530: Playwright-based link extraction (exactly as the real task does it).
GeoConfirmed: real REST API calls, verifies ≥1 video incident returned.

Run from repo root:
    cd scraper-engine && python ../test_scrape_live.py
"""
import sys
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_scrape_live")

SCRAPER_ENGINE_DIR = os.path.join(os.path.dirname(__file__), "scraper-engine")
sys.path.insert(0, SCRAPER_ENGINE_DIR)


# ── Funker530 ─────────────────────────────────────────────────────────

def test_funker530() -> None:
    logger.info("=" * 60)
    logger.info("TEST: Funker530 — Playwright link discovery (1 page)")
    logger.info("=" * 60)

    import asyncio
    from playwright.async_api import async_playwright
    from bs4 import BeautifulSoup
    from urllib.parse import urljoin, urlparse
    from config import settings

    base = settings.FUNKER530_BASE_URL

    async def scrape_one_page():
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1920, "height": 1080},
            )
            page = await context.new_page()
            try:
                url = f"{base}/videos/"
                logger.info(f"Navigating to {url}")
                await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                await page.wait_for_timeout(2000)
                html = await page.content()
                logger.info(f"Page HTML size: {len(html)} bytes")

                soup = BeautifulSoup(html, "lxml")
                post_links = []
                for a_tag in soup.find_all("a", href=True):
                    href = a_tag["href"]
                    full_url = urljoin(base, href)
                    parsed = urlparse(full_url)
                    if (
                        "funker530.com" in parsed.netloc
                        and len(parsed.path.strip("/").split("/")) >= 1
                        and not any(
                            skip in parsed.path
                            for skip in ["/category/", "/tag/", "/page/", "/author/"]
                        )
                        and parsed.path not in ["/", "/videos/", "/ukraine/"]
                        and parsed.path.strip("/")
                    ):
                        post_links.append(full_url)
                return list(dict.fromkeys(post_links))
            finally:
                await context.close()
                await browser.close()

    links = asyncio.run(scrape_one_page())
    logger.info(f"Funker530: found {len(links)} candidate post links")
    for u in links[:5]:
        logger.info(f"  {u}")

    assert len(links) > 0, f"Funker530: expected ≥1 post link — got 0"
    logger.info("PASS: Funker530\n")


# ── GeoConfirmed ──────────────────────────────────────────────────────

def test_geoconfirmed() -> None:
    logger.info("=" * 60)
    logger.info("TEST: GeoConfirmed — REST API video incident fetch")
    logger.info("=" * 60)

    from tasks.scrape_geoconfirmed import extract_video_incidents

    incidents = extract_video_incidents(max_incidents=5)
    logger.info(f"GeoConfirmed: fetched {len(incidents)} video incidents")
    for inc in incidents:
        logger.info(
            f"  [{inc['url_hash'][:8]}] {inc['url'][:80]}  "
            f"title={inc['title'][:50]!r}"
        )

    assert len(incidents) > 0, "GeoConfirmed: expected ≥1 video incident — got 0"
    logger.info("PASS: GeoConfirmed\n")


# ── Runner ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    passed: list[str] = []
    failed: list[str] = []

    for name, fn in [("funker530", test_funker530), ("geoconfirmed", test_geoconfirmed)]:
        try:
            fn()
            passed.append(name)
        except Exception as exc:
            logger.error(f"FAIL: {name} — {exc}", exc_info=True)
            failed.append(name)

    logger.info("=" * 60)
    logger.info(f"Results: {len(passed)} passed, {len(failed)} failed")
    if failed:
        logger.error(f"Failed: {failed}")
        sys.exit(1)
    else:
        logger.info("All tests passed!")
