"""
Borrowed from: https://dspy.ai/tutorials/mem0_react_agent/
"""

import dspy
from mem0 import Memory
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from api_keys import TOGETHER_API_KEY
import datetime
import time

# Configure environment
os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY

config = {
    "llm": {
        "provider": "together",
        "config": {
            "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "temperature": 0.1
        }
    },
    "embedder": {
        "provider": "together",
        "config": {
            "model": "Alibaba-NLP/gte-modernbert-base"
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
      "embedding_model_dims": 768
        }
    }
}


class MemoryTools:
    """Tools for interacting with the Mem0 memory system."""

    def __init__(self, memory: Memory):
        self.memory = memory

    def store_memory(self, content: str, user_id: str = "default_user") -> str:
         """
        Store a piece of information in memory.

        This is typically called when the agent learns something that should
        persist across turns (e.g., user preferences, reminders, personal facts).

        Args:
            content (str): The text to store in memory.
            user_id (str): Identifier for the user whose memory this belongs to.

        Returns:
            str: A confirmation message or an error message.
        """
        try:
            self.memory.add(content, user_id=user_id) # Add the content to Mem0's memory store.
            return f"Stored memory: {content}"
        except Exception as e:
            return f"Error storing memory: {str(e)}"

    def search_memories(self, query: str, user_id: str = "default_user", limit: int = 5) -> str:
        """
        Search memory for items relevant to a query.

        This is used when the agent needs to recall previously stored information,
        such as user preferences or earlier statements.

        Args:
            query (str): Natural-language search query.
            user_id (str): Identifier for the user whose memory should be searched.
            limit (int): Maximum number of memories to return.

        Returns:
            str: A formatted list of relevant memories or a message indicating
            that nothing was found.
        """
        try:
            # TODO: search for relevant memories; for your reference, it would be helpful to read the documentation of 
            # mem0 to see how to use the search method: https://github.com/mem0ai/mem0
            results = #TODO
            if not results:
                return "No relevant memories found."
            # Format the retrieved memories into a readable text block
            memory_text = "Relevant memories found:\n"
            for i, result in enumerate(results["results"]):
                memory_text += f"{i}. {result['memory']}\n"
            return memory_text
        except Exception as e:
            return f"Error searching memories: {str(e)}"

    def get_all_memories(self, user_id: str = "default_user") -> str:
        """Get all memories for a user."""
        try:
            results = self.memory.get_all(user_id=user_id)   # Fetch all memories associated with the given user
            if not results:
                return "No memories found for this user."

            memory_text = "All memories for user:\n"
            for i, result in enumerate(results["results"]):
                memory_text += f"{i}. {result['memory']}\n"
            return memory_text
        except Exception as e:
            return f"Error retrieving memories: {str(e)}"

    def update_memory(self, memory_id: str, new_content: str) -> str:
        """Update an existing memory."""
        try:
            self.memory.update(memory_id, new_content)    # Replace the old memory content with the new content
            return f"Updated memory with new content: {new_content}"
        except Exception as e:
            return f"Error updating memory: {str(e)}"

    def delete_memory(self, memory_id: str) -> str:
        """Delete a specific memory."""
        try:
            self.memory.delete(memory_id)
            return "Memory deleted successfully."
        except Exception as e:
            return f"Error deleting memory: {str(e)}"

def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class MemoryQA(dspy.Signature):
    """
    TODO: write the goal of the agent and define the input and output
    """
    user_input: #TODO
    response: #TODO

class MemoryReActAgent(dspy.Module):
    """A ReAct agent enhanced with Mem0 memory capabilities."""

    def __init__(self, memory: Memory):
        super().__init__()
        self.memory_tools = MemoryTools(memory)

        # Create tools list for ReAct
        #TODO: define the tools list
        # As a hint, you should consider all the functions provided in the MemoryTools class
        self.tools = #TODO

        # Initialize ReAct with our tools
        self.react = dspy.ReAct(
            signature=MemoryQA,
            tools=self.tools,
            max_iters=6
        )

    def forward(self, user_input: str):
        """Process user input with memory-aware reasoning."""

        return self.react(user_input=user_input)

    def set_reminder(self, reminder_text: str, date_time: str = None, user_id: str = "default_user") -> str:
        """Set a reminder for the user."""
        reminder = f"Reminder set for {date_time}: {reminder_text}"
        return self.memory_tools.store_memory(
            f"REMINDER: {reminder}", 
            user_id=user_id
        ) # Store the reminder as a memory entry.

    def get_preferences(self, category: str = "general", user_id: str = "default_user") -> str:
        """Get user preferences for a specific category."""
        query = f"user preferences {category}"
        return self.memory_tools.search_memories(
            query=query,
            user_id=user_id
        )  # Search memory for entries related to this category

    def update_preferences(self, category: str, preference: str, user_id: str = "default_user") -> str:
        """Update user preferences."""
        preference_text = f"User preference for {category}: {preference}"
        return self.memory_tools.store_memory(
            preference_text,
            user_id=user_id
        )  # Store the preference in memory so it can be retrieved later

def run_memory_agent_demo():
    """Demonstration of memory-enhanced ReAct agent."""

    # Configure DSPy
    lm = dspy.LM(model='together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1')
    dspy.configure(lm=lm)

    # Initialize memory system
    memory = Memory.from_config(config)

    # Create our agent
    agent = MemoryReActAgent(memory)

    # Sample conversation demonstrating memory capabilities
    print("🧠 Memory-Enhanced ReAct Agent Demo")
    print("=" * 50)

    conversations = [
        # TODO: add sample user inputs to demonstrate the agent's memory capabilities
        # prompt1: this prompt should reveal certain information 
        # prompt2: reveal more information
        # prompt3: ask the agent to recall previously stored information
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
    run_memory_agent_demo()
