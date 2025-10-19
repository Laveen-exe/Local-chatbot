from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
import time
import requests
from google.auth.transport.requests import Request
from google.oauth2 import service_account
import sqlite3
import streamlit as st
import json

def generate_content_with_text(prompt):
    """
    Sends a text prompt to the Gemini model and returns the response and response time.
    """
    start_time = time.time()

    try:
        # Get token from service account
        json_str = st.secrets["GEMINI_SERVICE_ACCOUNT_JSON"]
        creds_dict = json.loads(json_str)
        creds = service_account.Credentials.from_service_account_info(
            creds_dict,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        creds.refresh(Request())
        # API details - using the non-streaming endpoint for a complete response
        url = f"https://aiplatform-genai1.p.googleapis.com/v1/projects/prj-shared-connectivity-rg-001/locations/global/publishers/google/models/gemini-2.5-flash:generateContent"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {creds.token}"
        }
        print("Headers:", headers)
        # Request payload (text only)
        data = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.4, "topP": 1.0, "topK": 32, "maxOutputTokens": 4000}
        }

        # Make the request
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        json_response = response.json()
        response_text = ""
        if "candidates" in json_response and json_response["candidates"]:
            candidate = json_response["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                for part in candidate["content"]["parts"]:
                    if "text" in part:
                        response_text += part["text"]
        
        response_time = time.time() - start_time
        return {"response": response_text, "response_time": response_time}

    except Exception as e:
        response_time = time.time() - start_time
        error_message = f"An error occurred: {e}"
        if 'response' in locals() and hasattr(response, 'text'):
            error_message += f" | Response content: {response.text}"
        return {"error": error_message, "response_time": response_time}

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    
    # Convert messages to a single prompt string for Gemini
    prompt_parts = []
    for message in messages:
        if isinstance(message, HumanMessage):
            prompt_parts.append(f"User: {message.content}")
        elif isinstance(message, AIMessage):
            prompt_parts.append(f"Assistant: {message.content}")
    
    # Create the full prompt
    prompt = "\n".join(prompt_parts)
    
    # Get response from Gemini
    result = generate_content_with_text(prompt)
    
    if "error" in result:
        response_content = f"Error: {result['error']}"
    else:
        response_content = result['response']
    
    # Create AIMessage for response
    response = AIMessage(content=response_content)
    return {"messages": [response]}

conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
# Checkpointer
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads = set()
    try:
        # Try to list all checkpoints. If API requires a config, use an empty thread_id.
        for checkpoint in checkpointer.list({"configurable": {"thread_id": ""}}):
            # Defensive: Only add if thread_id exists in config
            thread_id = checkpoint.config.get('configurable', {}).get('thread_id')
            if thread_id:
                all_threads.add(thread_id)
    except Exception as e:
        # If listing fails, return empty list
        print(f"Error listing threads: {e}")
    return list(all_threads)
