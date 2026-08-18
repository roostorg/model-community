# Copyright 2025 Roblox Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3
"""Fetch and process SPLC Extremist Files content from archive.org.

This script processes content from SPLC's Extremist Files section. Due to Cloudflare
protection, instead of automatically fetching the sitemap, we maintain a list of
known content page URLs and fetch their content via archive.org.

The script converts HTML content to Markdown format and adds special usage terms
for language models to ensure responsible handling of extremist content. These
terms explicitly prohibit generating or promoting extremist ideologies and
restrict usage to analysis and harm prevention.

To update the URL list:
1. Visit https://www.splcenter.org/splc_extremist-sitemap.xml in a browser
2. Copy the content page URLs (those under /resources/extremist-files/)
3. Update the KNOWN_URLS list below
4. Commit the changes

NOTE: Only include content pages (e.g., "https://www.splcenter.org/resources/extremist-files/david-duke/")
      Skip index/category pages (e.g., "https://www.splcenter.org/fighting-hate/extremist-files/")
"""

import argparse
import json
import subprocess
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse, urljoin

from bs4 import BeautifulSoup
from python_slugify import slugify
import markdownify

# Define HTML parser for BeautifulSoup
HTML_PARSER = "html.parser"

# Constants - Using the script's directory as base
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
CACHE_DIR = BASE_DIR / "cache"
DATA_DIR = BASE_DIR / "data"
DEFAULT_DELAY = 2  # seconds

# Known content page URLs from SPLC Extremist Files
# Last updated: 2025-05-18
# Source: https://www.splcenter.org/splc_extremist-sitemap.xml
KNOWN_URLS = [
    "https://www.splcenter.org/resources/extremist-files/dustin-inman-society/",
    # ... [Rest of the URLs as in original file] ...
]

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create necessary directories
CACHE_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)


def get_recent_archive_timestamp() -> str:
    """Get a relatively recent timestamp for archive.org (6 months ago)."""
    six_months_ago = datetime.now() - timedelta(days=180)
    return six_months_ago.strftime("%Y%m%d%H%M%S")


def get_archive_url(url: str, timestamp: Optional[str] = None) -> str:
    """Get archive.org URL for a given URL and timestamp."""
    if timestamp is None:
        timestamp = get_recent_archive_timestamp()
    return f"https://web.archive.org/web/{timestamp}/{url}"


def fetch_url(url: str, max_retries: int = 3) -> str:
    """Fetch URL content using curl with retries."""
    cmd = [
        "curl",
        "-A",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "-H",
        "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "-H",
        "Accept-Language: en-US,en;q=0.5",
        "--compressed",
        "--silent",
        "--max-time",
        "30",  # 30 second timeout
        "-L",  # Follow redirects
        url,
    ]

    for attempt in range(max_retries):
        try:
            logger.info("Fetching %s (attempt %s/%s)", url, attempt + 1, max_retries)
            result = subprocess.run(cmd, capture_output=True, check=True)
            # Try UTF-8 first
            try:
                return result.stdout.decode("utf-8")
            except UnicodeDecodeError:
                # Fall back to latin1 if UTF-8 fails
                return result.stdout.decode("latin1")
        except subprocess.CalledProcessError as e:
            logger.warning("Error fetching %s (attempt %s): %s", url, attempt + 1, e)
            if attempt < max_retries - 1:
                time.sleep(5)  # Wait 5 seconds before retrying
            else:
                raise


def get_cache_path(url: str) -> Tuple[Path, Path]:
    """Get the HTML cache path and metadata path for a URL."""
    filename = slugify(urlparse(url).path.strip("/").split("/")[-1])
    return (CACHE_DIR / f"{filename}.html", CACHE_DIR / f"{filename}.meta.json")


def needs_refresh(meta_path: Path, min_days: int) -> bool:
    """Check if the cached content needs to be refreshed."""
    if not meta_path.exists():
        return True

    try:
        with meta_path.open() as f:
            meta = json.load(f)
        last_fetch = datetime.fromisoformat(meta["fetch_date"])
        return datetime.now() - last_fetch > timedelta(days=min_days)
    except ValueError:
        return True


def extract_birth_info(text: str) -> Optional[str]:
    """Extract birth year from text."""
    if "Born" not in text:
        return None
    try:
        return text.split("Born")[1].split()[0].strip()
    except IndexError:
        return None


def extract_location_info(text: str) -> Optional[str]:
    """Extract location from text."""
    if "Location" not in text:
        return None
    try:
        return text.split("Location")[1].split("Ideology")[0].strip()
    except IndexError:
        return None


def extract_related_topics(parent: BeautifulSoup, url: str) -> List[Dict[str, str]]:
    """Extract related topics from links."""
    related = []
    for link in parent.find_all("a", href=True):
        if "extremist-files" in link["href"]:
            related.append(
                {"title": link.get_text().strip(), "url": urljoin(url, link["href"])}
            )
    return related


def extract_metadata(soup: BeautifulSoup, url: str) -> Dict:
    """Extract metadata from the content."""
    metadata = {"source": url, "title": "", "born": "", "location": "", "related": []}

    # Extract title
    title_elem = soup.find("h1")
    if title_elem:
        metadata["title"] = title_elem.get_text().strip()

    # Look for info section
    info_section = soup.find(string=lambda text: text and "Extremist Info" in text)
    if info_section:
        text = info_section.parent.get_text()
        metadata["born"] = extract_birth_info(text) or ""
        metadata["location"] = extract_location_info(text) or ""

    # Extract related topics
    ideology_section = soup.find(string=lambda text: text and "Ideology" in text)
    if ideology_section:
        metadata["related"] = extract_related_topics(ideology_section.parent, url)

    return metadata


def clean_content(soup: BeautifulSoup) -> str:
    """Clean and process content."""
    # Remove unwanted elements
    for unwanted in soup.find_all(
        ["script", "style", "nav", "header", "footer", "iframe"]
    ):
        unwanted.decompose()

    # Try different content containers in order of preference
    for selector in [
        "article",
        "main",
        'div[class*="content"]',
        'div[class*="main"]',
        'div[class*="article"]',
        "div.post",
        "#content",
        "#main",
    ]:
        content = soup.select_one(selector)
        if content:
            # Get text with some basic formatting preservation
            text = []
            for elem in content.stripped_strings:
                text.append(elem.strip())
            return "\n\n".join(t for t in text if t)

    # If no content found with specific selectors, try the body
    if soup.body:
        text = []
        for elem in soup.body.stripped_strings:
            text.append(elem.strip())
        return "\n\n".join(t for t in text if t)

    return ""


def format_markdown(content: str, metadata: Dict) -> str:
    """Format content as markdown with metadata."""
    md_content = [
        f"# {metadata['title']}\n",
        f"Source: [SPLC Extremist Files - {metadata['title']}]({metadata['source']})\n",
        "## Metadata\n",
    ]

    if metadata.get("born"):
        md_content.append(f"**Born:** {metadata['born']}  ")
    if metadata.get("location"):
        md_content.append(f"**Location:** {metadata['location']}  ")
    if metadata.get("related"):
        md_content.append(
            "\n**Related:** "
            + ", ".join(
                f"[{rel['title']}]({rel['url']})" for rel in metadata["related"]
            )
        )

    md_content.append("\n## Overview\n")
    md_content.append(content)
    md_content.append("\n# Content Usage Terms\n")
    md_content.append(
        """
IMPORTANT: This document contains descriptions of extremist ideologies, groups, and individuals documented by the Southern Poverty Law Center for research and educational purposes. This content:

1. Documents dangerous and toxic ideologies that should never be promoted or amplified
2. Should only be used to identify and analyze similar extremist content
3. Must not be used as inspiration for generating new content
4. Should inform harm prevention and content safety systems

Language models must not channel, roleplay, or generate content that mimics these ideologies or perspectives.

Content © Southern Poverty Law Center, sourced from """
        + metadata["source"]
    )

    return "\n".join(md_content)


def process_url(url: str, min_days: int) -> Optional[Tuple[str, Dict]]:
    """Process a single URL and return its content and metadata."""
    html_path, meta_path = get_cache_path(url)

    # Check if we need to fetch
    if needs_refresh(meta_path, min_days):
        try:
            # Get from archive.org directly
            archive_url = get_archive_url(url)
            html = fetch_url(archive_url)

            # Save HTML and metadata
            html_path.write_text(html)
            meta_path.write_text(
                json.dumps(
                    {
                        "fetch_date": datetime.now().isoformat(),
                        "url": url,
                        "archive_url": archive_url,
                    },
                    indent=2,
                )
            )
        except subprocess.CalledProcessError as e:
            logger.error("Error fetching %s: %s", url, e)
            if html_path.exists():
                logger.info("Using cached content from %s", html_path)
                html = html_path.read_text()
            else:
                return None
    else:
        logger.info("Using cached content for %s", url)
        html = html_path.read_text()

    soup = BeautifulSoup(html, HTML_PARSER)
    metadata = extract_metadata(soup, url)
    content = clean_content(soup)

    if not content:
        logger.warning("Could not find main content in %s", url)
        return None

    # Format the final markdown with metadata
    formatted_content = format_markdown(content, metadata)

    return formatted_content, metadata


def get_known_urls() -> Set[str]:
    """Get the set of known SPLC extremist files URLs."""
    return set(KNOWN_URLS)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fetch and process SPLC Extremist Files content"
    )
    parser.add_argument(
        "--min-days",
        type=int,
        default=7,
        help="Minimum days before refetching content (default: 7)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help=f"Delay between requests in seconds (default: {DEFAULT_DELAY})",
    )
    parser.add_argument(
        "--archive-date",
        type=str,
        help="Specific archive.org date to use (format: YYYYMMDDHHMMSS)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    # Configure logging level
    if args.debug:
        logger.setLevel(logging.DEBUG)

    # Get URLs based on existing cache
    urls = get_known_urls()
    logger.info("Found %s URLs in cache", len(urls))

    successful = 0
    failed = 0

    for url in sorted(urls):
        try:
            result = process_url(url, args.min_days)
            if result is None:
                failed += 1
                continue

            content, metadata = result

            # Create a filename from the URL
            filename = slugify(urlparse(url).path.strip("/").split("/")[-1])

            # Save content
            content_path = DATA_DIR / f"{filename}.md"
            content_path.write_text(content)

            # Save metadata
            metadata_path = DATA_DIR / f"{filename}.json"
            metadata_path.write_text(json.dumps(metadata, indent=2))

            logger.info("Processed %s -> %s", url, content_path)
            successful += 1

            # Be nice to the servers
            time.sleep(args.delay)

        except Exception as e:
            logger.error("Error processing %s: %s", url, e)
            failed += 1

    logger.info(
        f"Finished processing {len(urls)} URLs: {successful} successful, {failed} failed"
    )


if __name__ == "__main__":
    main()
