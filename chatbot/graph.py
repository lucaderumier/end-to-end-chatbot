from IPython.display import Image, display
import logging
from typing import Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START
from langchain_core.runnables import RunnableConfig

from chatbot.nodes import ChatbotNode, HumanInputNode
from chatbot.state import ChatbotState


LOG = logging.getLogger(__name__)

class ChatbotGraph:
    def __init__(self, model_name: str, temperature: float = 0):
        self.model_name = model_name
        self.temperature = temperature

        # Nodes
        self.system_prompt = "You are a helpful assistant."
        self.chatbot_node = ChatbotNode(self.model_name, self.temperature, self.system_prompt)
        self.human_input_node = HumanInputNode(self.model_name, self.temperature)

        # Memory saver
        self.memory_saver = MemorySaver()

    @property
    def graph(self):
        if hasattr(self, "_graph"):
            return self._graph
        self._graph = self.graph_builder()
        return self._graph

    def graph_builder(self):
        graph_builder = StateGraph(ChatbotState)

        graph_builder.add_node("chatbot_agent", self.chatbot_node.run)
        graph_builder.add_node("human_input", self.human_input_node.run)

        # Edges
        graph_builder.add_edge(START, "chatbot_agent")
        graph_builder.add_edge("human_input", "chatbot_agent")
        return graph_builder.compile(checkpointer=self.memory_saver)
    
    def display(self):
        display(Image(self.graph.get_graph(xray=True).draw_mermaid_png()))

    def invoke(self, input: str, config: Optional[RunnableConfig] = None):
        return self.graph.invoke(input, config)
    
