import os

from dotenv import load_dotenv

load_dotenv()

_client = None

HEALTH_DOMAINS = [
    "nih.gov",
    "pubmed.ncbi.nlm.nih.gov",
    "who.int",
    "cdc.gov",
    "medlineplus.gov",
    "realfood.gov",
    "usda.gov",
]


def _get_client():
    """Create the Tavily client on first search."""
    global _client
    if _client is None:
        from tavily import TavilyClient

        api_key = os.environ.get("TAVILY_API_KEY", "").strip()
        if not api_key:
            raise ValueError("Set TAVILY_API_KEY")
        _client = TavilyClient(api_key=api_key)
    return _client


def search_web(
    query: str,
    include_domains: list[str] | None = None,
) -> list[dict]:
    """Search allowed health sites and return their sources."""

    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")

    response = _get_client().search(
        query=query,
        search_depth="advanced",
        max_results=5,
        include_raw_content=False,
        include_domains=list(HEALTH_DOMAINS if include_domains is None else include_domains),
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
