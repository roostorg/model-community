"""Web-related tools."""

from __future__ import annotations

from typing import Dict, Any, Optional
import logging
import asyncio

from policy_cli.tools.base import BaseTool
from policy_cli.core.types import ToolResult, SimpleToolResult

logger = logging.getLogger(__name__)


class WebFetchTool(BaseTool):
    """Tool for fetching web content."""

    def __init__(self):
        super().__init__(
            name="web_fetch",
            description="Fetch content from a URL and return the text."
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to fetch content from"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default: 30)",
                    "default": 30
                },
                "max_length": {
                    "type": "integer",
                    "description": "Maximum content length to return (default: 50000)",
                    "default": 50000
                },
                "include_headers": {
                    "type": "boolean",
                    "description": "Include response headers in output",
                    "default": False
                }
            },
            "required": ["url"]
        }

    async def execute(
        self,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        url = parameters["url"]
        timeout = parameters.get("timeout", 30)
        max_length = parameters.get("max_length", 50000)
        include_headers = parameters.get("include_headers", False)

        try:
            import httpx

            # Validate URL
            if not url.startswith(("http://", "https://")):
                return SimpleToolResult(
                    content=f"Invalid URL: {url}. Must start with http:// or https://",
                    error="Invalid URL"
                )

            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(url, follow_redirects=True)
                response.raise_for_status()

                # Get content
                content = response.text

                # Truncate if too long
                if len(content) > max_length:
                    content = content[:max_length] + f"\n\n[Content truncated at {max_length} characters]"

                # Format result
                result_parts = []

                if include_headers:
                    result_parts.append("Response Headers:")
                    for key, value in response.headers.items():
                        result_parts.append(f"  {key}: {value}")
                    result_parts.append("")

                result_parts.append(f"Content from {url}:")
                result_parts.append(content)

                full_content = "\n".join(result_parts)

                return SimpleToolResult(
                    content=full_content,
                    display_content=f"Fetched {len(response.text)} characters from {url}"
                )

        except Exception as e:
            return SimpleToolResult(
                content=f"Failed to fetch URL: {str(e)}",
                error=str(e)
            )


class WebSearchTool(BaseTool):
    """Tool for web search (simplified implementation)."""

    def __init__(self):
        super().__init__(
            name="web_search",
            description="Search the web for information. Returns search results and snippets."
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (default: 5)",
                    "default": 5
                }
            },
            "required": ["query"]
        }

    async def execute(
        self,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        query = parameters["query"]
        num_results = parameters.get("num_results", 5)

        try:
            # This is a simplified implementation
            # In a real implementation, you'd integrate with search APIs like:
            # - Google Custom Search API
            # - Bing Search API
            # - DuckDuckGo API
            # - SerpAPI

            import httpx
            import urllib.parse

            # Use DuckDuckGo Instant Answers API as a simple example
            encoded_query = urllib.parse.quote(query)
            url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&no_html=1&skip_disambig=1"

            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()

                # Format results
                results = []

                # Abstract (direct answer)
                if data.get("Abstract"):
                    results.append(f"**Answer:** {data['Abstract']}")
                    if data.get("AbstractURL"):
                        results.append(f"**Source:** {data['AbstractURL']}")

                # Related topics
                if data.get("RelatedTopics"):
                    results.append("\n**Related Topics:**")
                    for i, topic in enumerate(data["RelatedTopics"][:num_results]):
                        if isinstance(topic, dict) and topic.get("Text"):
                            results.append(f"{i+1}. {topic['Text']}")
                            if topic.get("FirstURL"):
                                results.append(f"   URL: {topic['FirstURL']}")

                if not results:
                    content = f"No direct results found for query: {query}\n\nThis is a simplified search implementation. For comprehensive web search, consider integrating with Google Custom Search API or similar services."
                else:
                    content = f"Search results for: {query}\n\n" + "\n".join(results)

                return SimpleToolResult(
                    content=content,
                    display_content=f"Searched for: {query}"
                )

        except Exception as e:
            # Fallback message
            return SimpleToolResult(
                content=f"Web search not available: {str(e)}\n\nTo enable full web search functionality, configure a search API (Google Custom Search, Bing, etc.) in your settings.",
                error=str(e)
            )


class UrlAnalyzerTool(BaseTool):
    """Tool for analyzing URLs and extracting metadata."""

    def __init__(self):
        super().__init__(
            name="analyze_url",
            description="Analyze a URL and extract metadata like title, description, and key information."
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to analyze"
                },
                "extract_links": {
                    "type": "boolean",
                    "description": "Extract links from the page",
                    "default": False
                }
            },
            "required": ["url"]
        }

    async def execute(
        self,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        url = parameters["url"]
        extract_links = parameters.get("extract_links", False)

        try:
            import httpx
            from html.parser import HTMLParser
            import re

            class MetaExtractor(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.title = ""
                    self.description = ""
                    self.keywords = ""
                    self.links = []
                    self.in_title = False

                def handle_starttag(self, tag, attrs):
                    attrs_dict = dict(attrs)

                    if tag == "title":
                        self.in_title = True
                    elif tag == "meta":
                        name = attrs_dict.get("name", "").lower()
                        property_attr = attrs_dict.get("property", "").lower()
                        content = attrs_dict.get("content", "")

                        if name == "description" or property_attr == "og:description":
                            self.description = content
                        elif name == "keywords":
                            self.keywords = content
                        elif property_attr == "og:title" and not self.title:
                            self.title = content
                    elif tag == "a" and extract_links:
                        href = attrs_dict.get("href")
                        if href:
                            self.links.append(href)

                def handle_endtag(self, tag):
                    if tag == "title":
                        self.in_title = False

                def handle_data(self, data):
                    if self.in_title:
                        self.title += data.strip()

            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(url, follow_redirects=True)
                response.raise_for_status()

                # Parse HTML
                parser = MetaExtractor()
                parser.feed(response.text)

                # Build result
                result_parts = [f"URL Analysis: {url}"]

                if response.url != url:
                    result_parts.append(f"Final URL: {response.url}")

                result_parts.append(f"Status Code: {response.status_code}")
                result_parts.append(f"Content Type: {response.headers.get('content-type', 'unknown')}")

                if parser.title:
                    result_parts.append(f"Title: {parser.title}")

                if parser.description:
                    result_parts.append(f"Description: {parser.description}")

                if parser.keywords:
                    result_parts.append(f"Keywords: {parser.keywords}")

                # Content length
                content_length = len(response.text)
                result_parts.append(f"Content Length: {content_length:,} characters")

                # Extract some key info from content
                text_content = re.sub(r'<[^>]+>', ' ', response.text)
                text_content = re.sub(r'\s+', ' ', text_content).strip()

                if len(text_content) > 500:
                    result_parts.append(f"Content Preview: {text_content[:500]}...")
                else:
                    result_parts.append(f"Content Preview: {text_content}")

                if extract_links and parser.links:
                    result_parts.append(f"\nLinks found ({len(parser.links)}):")
                    for i, link in enumerate(parser.links[:10]):  # Limit to first 10
                        result_parts.append(f"  {i+1}. {link}")
                    if len(parser.links) > 10:
                        result_parts.append(f"  ... and {len(parser.links) - 10} more")

                content = "\n".join(result_parts)

                return SimpleToolResult(
                    content=content,
                    display_content=f"Analyzed URL: {url}"
                )

        except Exception as e:
            return SimpleToolResult(
                content=f"Failed to analyze URL: {str(e)}",
                error=str(e)
            )