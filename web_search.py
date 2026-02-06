import os
import requests
from typing import Optional
from serpapi import GoogleSearch
from bs4 import BeautifulSoup
import dspy
from api_keys import TOGETHER_API_KEY, SERPAPI_API_KEY
import time

os.environ["SERPAPI_API_KEY"] =  SERPAPI_API_KEY
os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY


def extract_text(html: str) -> str:
    """
    Extracts clean, readable text from raw HTML.

    This function takes an HTML page (returned from a web request) and removes
    non-readable content. 

    Args:
        html (str): Raw HTML content of a web page.

    Returns:
        str: Cleaned plain-text version of the page content.
    """
    soup = BeautifulSoup(html, "html.parser")   # Parse the raw HTML into a structured BeautifulSoup object
    for tag in soup(["script", "style", "noscript"]): # Remove tags that do not contain meaningful visible text
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)  # Extract all remaining visible text from the HTML
    return " ".join(text.split())


class WebTools:
    """
    Utility class that provides web search and page-reading tools for the agent.

    This class acts as a thin wrapper around a web search API (SerpAPI).
    The agent calls these methods when it needs *external information*
    that is not available in its prompt or memory.

    Conceptually:
    - The LLM decides *when* to search
    - This class handles *how* the search is executed
    - The returned text is formatted to be readable by both humans and LLMs
    """

    def __init__(self, serpapi_key: Optional[str] = None):
        self.serpapi_key = serpapi_key or os.getenv("SERPAPI_API_KEY")

    def web_search(self, query: str, num_results: int = 5, page: int = 1) -> str:
        """
        Search the web and return top links/snippets.
        Args:
          query: search query string
          num_results: number of results to return (recommended <= 10)
          page: pagination index starting from 1
        Returns:
          A formatted list of results with title, link, and snippet.
        """
        if not self.serpapi_key:
            return "Error: SERPAPI_API_KEY is not set."

        # Bing via SerpAPI. Pagination is controlled by 'first' for bing.
        # page=1 => first=0, page=2 => first=num_results, etc.
        first = (max(page, 1) - 1) * num_results
        # Parameters passed to the SerpAPI search endpoint.
        # We use Bing here, but SerpAPI supports multiple search engines.
        params = {
            "engine": "bing",
            "q": query,
            "api_key": self.serpapi_key,
            "count": num_results,
            "first": first,
        }

        try:
            results = GoogleSearch(params).get_dict()  # Execute the search request and parse the JSON response
            organic = results.get("organic_results", []) or []
            if not organic:
                return "No results found."
            # Build a human- and LLM-readable summary of the results
            lines = [f"Web search results for: {query} (page {page})"]
            for i, item in enumerate(organic[:num_results], 1):
                title = item.get("title") or "(no title)"
                link = item.get("link") or "(no link)"
                snippet = item.get("snippet") or ""
                # Each result is formatted as a numbered block
                lines.append(f"{i}. {title}\n   {link}\n   {snippet}".strip())

            return "\n".join(lines)

        except Exception as e:
            return f"Error during web_search: {str(e)}"

class WebSearchQA(dspy.Signature):
    """
    [TODO]: define the objective of the web search agent.
    """
    user_input: str = #TODO
    response: str = #TODO

class WebSearchAgent(dspy.Module):
    """A ReAct agent enhanced with web search capabilities."""

    def __init__(self):
        super().__init__()
        # TODO: define self.web_tools (uncomment and complete the line below)
        #self.web_tools = 

        # TODO: define self.tools (uncomment and complete the line below)
        # self.tools = []

        self.react = dspy.ReAct(
            signature=WebSearchQA,
            tools=self.tools,
            max_iters=6
        )

    def forward(self, user_input: str):
        """Process user input."""
        # TODO: write the correct return value
        



def run_web_search_agent_demo():
    """Demonstration of web search ReAct agent."""

    # Configure DSPy
    lm = dspy.LM(model='together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1')
    dspy.configure(lm=lm)

    # Create our agent
    agent = WebSearchAgent()

    # Sample conversation demonstrating memory capabilities
    print("Web Search ReAct Agent Demo")
    print("=" * 50)

    conversations = [
        "What is the latest movie in December, 2025?",
        "What are some movies from this month that I should watch?"
        # TODO: Write 5 more prompts to demonstrate the web search capabilities of the agent.
        # PROMPT1: 
        # PROMPT2:
        # PROMPT3:
        # PROMPT4:
        # PROMPT5:
    ]

    for i, user_input in enumerate(conversations, 1):
        print(f"\n📝 User: {user_input}")

        try:
            response = agent(user_input=user_input)
            print(f"🤖 Agent: {response.response}")
            time.sleep(1)

        except Exception as e:
            print(f"❌ Error: {e}")

# Run the demonstration
if __name__ == "__main__":
    run_web_search_agent_demo()
