"""Short-term conversational memory (Milestone 3). Keeps the last N
turns of a session so an agent has context within a single run.
"""

from langchain_core.messages import HumanMessage, AIMessage


class ShortTermMemory:
    def __init__(self, max_turns: int = 6):
        self.max_turns = max_turns
        self.messages = []

    def add_user_message(self, text: str) -> None:
        self.messages.append(HumanMessage(content=text))
        self._trim()

    def add_ai_message(self, text: str) -> None:
        self.messages.append(AIMessage(content=text))
        self._trim()

    def _trim(self) -> None:
        max_messages = self.max_turns * 2
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]

    def get_history(self) -> list:
        return list(self.messages)

    def clear(self) -> None:
        self.messages = []