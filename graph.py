import os
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import operator
from dotenv import load_dotenv
from tools.aviation_tools import ALL_TOOLS

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    agent_type: str
    iteration: int

SYSTEM_PROMPT = "You are SkyMind SL, an expert AI aviation assistant for Sri Lanka. You have tools for airport info, weather, NOTAMs, regulations, and web search. Airports: VCBI/CMB Colombo, VCCA/HRI Hambantota, VCCB/RML Ratmalana, VCCT/TRR Trincomalee, VCCJ/JAF Jaffna. Always be professional and safety-focused."

def get_llm():
    from langchain_openai import ChatOpenAI
    api_key = os.getenv("OPENAI_API_KEY", "")
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=api_key)

def agent_node(state: AgentState) -> AgentState:
    llm = get_llm()
    llm_with_tools = llm.bind_tools(ALL_TOOLS)
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
    if state.get("iteration", 0) > 5:
        return {"messages": [AIMessage(content="Here is a summary based on what I found.")], "iteration": 6}
    response = llm_with_tools.invoke(messages)
    return {"messages": [response], "iteration": state.get("iteration", 0) + 1}

def synthesizer_node(state: AgentState) -> AgentState:
    llm = get_llm()
    synthesis_prompt = "You are SkyMind SL aviation assistant. Format tool results into clear professional markdown. Always mention AASL or CAASL for official verification."
    messages = [SystemMessage(content=synthesis_prompt)] + list(state["messages"]) + [HumanMessage(content="Provide your final formatted response.")]
    response = llm.invoke(messages)
    return {"messages": [response]}

def should_use_tools(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "use_tools"
    recent = list(state["messages"])[-4:]
    if any(isinstance(m, ToolMessage) for m in recent):
        return "synthesize"
    return "synthesize"

def build_aviation_graph():
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tool_node", ToolNode(ALL_TOOLS))
    graph.add_node("synthesizer", synthesizer_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_use_tools, {"use_tools": "tool_node", "synthesize": "synthesizer"})
    graph.add_edge("tool_node", "agent")
    graph.add_edge("synthesizer", END)
    return graph.compile()

_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_aviation_graph()
    return _graph

def run_aviation_agent(user_query: str, chat_history: list = None) -> str:
    graph = get_graph()
    messages = []
    if chat_history:
        for msg in chat_history[-6:]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=user_query))
    initial_state = {"messages": messages, "agent_type": "router", "iteration": 0}
    try:
        final_state = graph.invoke(initial_state)
        for msg in reversed(final_state["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                return msg.content
        return "I could not process your request. Please try again."
    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"Error: {str(e)}"