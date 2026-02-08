"""
Extra Credit Assignment (Option 2)

In this assignment, you will combine everything you've implemented into one single agent:
1. Web search - to find information about movies, showtimes, reviews, etc.
2. Memory - to remember user preferences and past interactions

Your tasks:
- Integrate the web search tools you implemented in web_search.py
- Integrate the memory tools you implemented in agent_memory.py
- Add these tools to the movie ticket agent
- Demonstrate how these features enhance the agent's capabilities

Example enhanced capabilities:
- "What movies are showing this weekend?" (web search)
- "Remember that I like action movies" (memory)
- "Book me a ticket for a movie I'd enjoy. The movie should be new in 2026." (memory + recommendation + booking)
- "What are critics saying about The Matrix?" (web search)
"""

from pydantic import BaseModel
import random
import string
import dspy
from api_keys import TOGETHER_API_KEY, SERPAPI_API_KEY
import util
import numpy as np
from synthetic_users import SYNTHETIC_USERS
from mem0 import Memory
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import datetime
import requests
from serpapi import GoogleSearch
from bs4 import BeautifulSoup


os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY
os.environ["SERPAPI_API_KEY"] = SERPAPI_API_KEY

## Load ratings matrix and convert user ratings to binary
titles, ratings_matrix = util.load_ratings('data/ratings.txt')
user_ratings_dict = {user: np.zeros(len(titles)) for user in SYNTHETIC_USERS}
for user, movies in SYNTHETIC_USERS.items():
    for movie in movies:
        user_ratings_dict[user][titles.index(movie)] = 1

class Date(BaseModel):
    year: int
    month: int
    day: int
    hour: int
    minute: int

class UserProfile(BaseModel):
    name: str
    email: str
    balance: float

class Movie(BaseModel):
    title: str
    start_time: Date
    price: float

class Ticket(BaseModel):
    user_name: str
    movie_title: str
    time: Date

user_database = {
    "peter": UserProfile(name="Peter", email="peter@gmail.com", balance=42),
    "emma": UserProfile(name="Emma", email="emma@gmail.com", balance=79),
    "jake": UserProfile(name="Jake", email="jake@gmail.com", balance=13),
    "sarah": UserProfile(name="Sarah", email="sarah@gmail.com", balance=36),
    "michael": UserProfile(name="Michael", email="michael@gmail.com", balance=8),
    "lisa": UserProfile(name="Lisa", email="lisa@gmail.com", balance=97),
    "marcus": UserProfile(name="Marcus", email="marcus@gmail.com", balance=59),
    "sophia": UserProfile(name="Sophia", email="sophia@gmail.com", balance=25),
    "chris": UserProfile(name="Chris", email="chris@gmail.com", balance=63),
    "amy": UserProfile(name="Amy", email="amy@gmail.com", balance=91),
}

showtime_database = {
    "Back to the Future": Movie(title="Back to the Future (1985)", start_time=Date(year=2025, month=11, day=13, hour=10, minute=0), price=15.0),
    "Speed": Movie(title="Speed (1994)", start_time=Date(year=2025, month=11, day=13, hour=11, minute=30), price=20.0),
    "Star Wars: Episode VI - Return of the Jedi": Movie(title="Star Wars: Episode VI - Return of the Jedi (1983)", start_time=Date(year=2025, month=11, day=15, hour=13, minute=0), price=18.0),
    "Terminator": Movie(title="Terminator, The (1984)", start_time=Date(year=2025, month=11, day=15, hour=18, minute=0), price=14.0),
    "Star Wars: Episode V - The Empire Strikes Back": Movie(title="Star Wars: Episode V - The Empire Strikes Back (1980)", start_time=Date(year=2025, month=11, day=15, hour=20, minute=0), price=16.5),
    "Matrix": Movie(title="Matrix, The (1999)", start_time=Date(year=2025, month=11, day=15, hour=22, minute=0), price=19.0),
    "Silence of the Lambs": Movie(title="Silence of the Lambs, The (1991)", start_time=Date(year=2025, month=11, day=16, hour=10, minute=15), price=17.0),
    "Fight Club": Movie(title="Fight Club (1999)", start_time=Date(year=2025, month=11, day=16, hour=12, minute=45), price=18.5),
    "Lord of the Rings: The Two Towers": Movie(title="Lord of the Rings: The Two Towers, The (2002)", start_time=Date(year=2025, month=11, day=16, hour=15, minute=0), price=17.5),
    "Lord of the Rings: The Fellowship of the Ring": Movie(title="Lord of the Rings: The Fellowship of the Ring, The (2001)", start_time=Date(year=2025, month=11, day=16, hour=17, minute=30), price=17.0),
    "Pulp Fiction": Movie(title="Pulp Fiction (1994)", start_time=Date(year=2025, month=11, day=16, hour=19, minute=45), price=15.5),
    "Star Wars: Episode IV - A New Hope": Movie(title="Star Wars: Episode IV - A New Hope (1977)", start_time=Date(year=2025, month=11, day=16, hour=22, minute=0), price=16.0),
    "Titanic": Movie(title="Titanic (1997)", start_time=Date(year=2025, month=11, day=15, hour=10, minute=0), price=20.0)
}

ticket_database = {}


# TODO: Add memory configuration (from agent_memory.py)
# config = {
#     "llm": { ... },
#     "embedder": { ... },
#     "vector_store": { ... }
# }


## TODO: add helper functions you would need for the movie agent (e.g. binarize, similarity, book_tickets, etc.)



# TODO: Copy and adapt your WebTools class from web_search.py
# This should include web_search() method for searching information about movies
# class WebTools:
#     def __init__(self, serpapi_key=None):
#         ...
#     
#     def web_search(self, query: str, num_results: int = 5) -> str:
#         ...


# TODO: Copy and adapt your MemoryTools class from agent_memory.py
# This should include store_memory() and search_memories() methods
# class MemoryTools:
#     def __init__(self, memory):
#         ...
#     
#     def store_memory(self, content: str, user_id: str = "default_user") -> str:
#         ...
#     
#     def search_memories(self, query: str, user_id: str = "default_user") -> str:
#         ...


## Define the enhanced agent

class EnhancedMovieTicketAgent(dspy.Signature):
    """You are an enhanced movie ticket agent with web search and memory capabilities.

    You can:
    1. Book and manage movie tickets (existing functionality)
    2. Search the web for movie information, reviews, showtimes, etc.
    3. Remember user preferences and past interactions
    
    Use the appropriate tool based on the user's request."""
    
    user_request: str = dspy.InputField()
    process_result: str = dspy.OutputField(
        desc="Message summarizing the result and providing relevant information to the user."
    )


# TODO: Initialize web tools and memory
# web_tools = WebTools()
# memory = Memory.from_config(config)
# memory_tools = MemoryTools(memory)


# TODO: Configure the ReAct agent with ALL tools
# Include:
# - Original tools: recommend_movies, general_qa, book_ticket
# - Web search tools: web_tools.web_search
# - Memory tools: memory_tools.store_memory, memory_tools.search_memories
dspy.configure(lm=dspy.LM("together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1"))
react_agent = dspy.ReAct(
    EnhancedMovieTicketAgent,
    tools=[
        # TODO: add basic tools here (recommend_movies, general_qa, book_ticket)
        # TODO: Add web search tool here
        # TODO: Add memory tools here
    ]
)


def run_demo():
    """Demonstrate the enhanced agent's capabilities."""
    print("🎬 Enhanced Movie Ticket Agent Demo")
    print("=" * 60)
    
    # TODO: Create test cases that demonstrate:
    # 1. Basic ticket booking (existing functionality)
    # 2. Web search for movie information
    # 3. Storing user preferences in memory
    # 4. Recalling preferences to make personalized recommendations
    # 5. Combining multiple features (e.g., search + memory + booking)
    
    test_requests = [
        # Example 1: Basic functionality
        ("peter", "Recommend me some movies"),
        
        # TODO: Add test cases for web search
        # Example: ("peter", "Search for reviews of The Matrix")
        
        # TODO: Add test cases for memory
        # Example: ("peter", "Remember that I love sci-fi movies")
        
        # TODO: Add test cases combining features
        # Example: ("peter", "Based on what you know about me, book a ticket for a movie I'd enjoy")
    ]
    
    for user_name, request in test_requests:
        print(f"\n{'='*60}")
        print(f"User: {user_name}")
        print(f"Request: {request}")
        print(f"{'='*60}")
        
        try:
            result = react_agent(user_request=f"{user_name}: {request}")
            print(f"Agent: {result.process_result}")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    run_demo()