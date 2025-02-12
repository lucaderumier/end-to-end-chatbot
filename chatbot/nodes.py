import logging
from typing import Optional, List

from langchain_core.messages import SystemMessage
from langgraph.types import interrupt

from chatbot.state import ChatbotState
from chatbot.utils.model_provider import get_model

LOG = logging.getLogger(__name__)

class ChatbotNode:
    def __init__(
        self, 
        model_name: str, 
        temperature: float = 0,
        system_prompt: str = "You are a helpful assistant."
    ):
        self.system_message = SystemMessage(content=system_prompt)
        self.model = get_model(model_name, temperature)


    def run(self, state: ChatbotState):
        messages = state["messages"]

        # Ensure system message is always first
        if not messages or type(messages[0]) != SystemMessage:
            messages.insert(0, self.system_message)

        response = self.model.invoke(messages)
        state["messages"].append(response)
        return state

class HumanInputNode:
    def __init__(self, model_name: str, temperature: float = 0):
        self.model = get_model(model_name, temperature)

    def run(self, state: ChatbotState):
        human_input = interrupt(f"{state['messages'][-1].content}\n\n")
        state["messages"].append(human_input)
        return state  # Hold state until user sends input