import os

from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

client = TavilyClient(
    api_key=os.environ["TAVILY_API_KEY"]
)


def search_web(query: str) -> list[dict]:
    """Search the web and return relevant medical sources."""

    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")

    response = client.search(
        query=query,
        search_depth="advanced",
        max_results=5,
        include_raw_content=False,
    )

    return response.get("results", [])


def format_search_results(results: list[dict]) -> str:
    """Format web search results as evidence for the LLM."""

    blocks = []

    for index, result in enumerate(results, start=1):
        title = result.get("title", "").strip()
        url = result.get("url", "").strip()
        content = result.get("content", "").strip()
        score = result.get("score")

        blocks.append(
            f"Source {index}\n"
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Relevance score: {score}\n"
            f"Content: {content}"
        )

    return "\n\n".join(blocks)


def format_source_links(results: list[dict]) -> str:
    """Format Tavily sources as clickable Markdown links."""

    sources = []

    for index, result in enumerate(results, start=1):
        title = result.get("title", "").strip()
        url = result.get("url", "").strip()

        if not url:
            continue

        if not title:
            title = url

        sources.append(
            f"- [{title}]({url})"
        )

    if not sources:
        return ""

    return "### Sources\n\n" + "\n".join(sources)
