from fastapi import FastAPI, HTTPException
import logging
from typing import Dict, Any
import traceback
import uuid

from langchain_core.messages import HumanMessage

from chatbot.graph import ChatbotGraph
from api.models import ChatRequest, ChatResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Chatbot API",
    description="API for interacting with Chatbot",
    version="1.0.0"
)

# Store chat sessions
chat_sessions: Dict[str, Dict[str, Any]] = {}

@app.post("/chat/start", response_model=ChatResponse)
async def start_chat(request: ChatRequest):
    """Start a new chat session with the Chatbot"""
    try:
        # Create new graph instance
        graph = ChatbotGraph(
            model_name=request.model_name,
            temperature=request.temperature,
        )

        # Generate a unique session ID
        session_id = str(uuid.uuid4())  
        
        # Invoke the chat
        inputs = {"messages": [{"role": "user", "content": request.user_input}]}
        config = {
            "recursion_limit": request.recursion_limit,
            "configurable": {"thread_id": session_id}
        }
        output = graph.invoke(
            input=inputs,
            config=config
        )

        # Store the graph instance
        chat_sessions[session_id] = {
            "graph": graph,
            "state": output,
            "config": config
        }
        
        return ChatResponse(
            response=output["messages"][-1].content,
            session_id=session_id
        )
    
    except Exception as e:
        logger.error(f"Error starting chat: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=traceback.format_exc())
    
# API: Continue chat with user input
@app.post("/chat/{session_id}/continue", response_model=ChatResponse)
async def continue_chat(session_id: str, request: ChatRequest):
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    # Retrieve session and add user message
    chat_session = chat_sessions[session_id]
    chat_session["state"]["messages"].append(HumanMessage(content=request.user_input))

    # Run chatbot response
    new_state = chat_session["graph"].invoke(chat_session["state"], chat_session["config"])

    # Update session
    chat_sessions[session_id].update({"state": new_state})

    return ChatResponse(
        response=new_state["messages"][-1].content,  # Last chatbot response
        session_id=session_id
    )

@app.get("/chat/sessions")
async def list_sessions():
    """List all active chat sessions"""
    return {"sessions": list(chat_sessions.keys())}

@app.delete("/chat/{session_id}")
async def end_session(session_id: str):
    """End a chat session"""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    del chat_sessions[session_id]
    return {"message": "Session ended successfully"}
