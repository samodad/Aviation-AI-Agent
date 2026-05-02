"""
LangGraph Diagram & Testing Script
Run this to visualize the agent graph and test individual components.

Usage: python agent/graph_diagram.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


def print_graph_structure():
    """Print a text representation of the LangGraph."""
    print("\n" + "="*60)
    print("  SKYMIND SL — LANGGRAPH AGENT ARCHITECTURE")
    print("="*60)
    print("""
  START
    │
    ▼
  ┌─────────────────────────────────────────────┐
  │          agent_node (GPT-4o-mini)            │
  │                                              │
  │  System Prompt: Aviation Expert for SL       │
  │  Tools Available: 5 aviation tools           │
  │  Pattern: ReAct (Reason → Act → Observe)     │
  └──────────────┬──────────────────────────────┘
                 │
          ┌──────▼──────┐
          │ Has tool     │
          │  calls?      │
          └──┬───────┬───┘
             │ YES   │ NO
             ▼       ▼
  ┌──────────────┐  ┌─────────────────────────┐
  │  tool_node   │  │    synthesizer_node      │
  │              │  │                          │
  │ • airport    │  │ Format results into      │
  │ • weather    │  │ clean Markdown response  │
  │ • notam      │  │ with aviation context    │
  │ • regs       │  │                          │
  │ • web search │  └────────────┬────────────┘
  └──────┬───────┘               │
         │                       ▼
         │ (loop back)         END
         ▼
     agent_node
  (processes tool results)

  Conditional Edges:
  • agent → tool_node   (when tool_calls present)
  • agent → synthesizer (when tool results available OR no tools needed)
  • tool_node → agent   (always, to process results)
  • synthesizer → END   (always)
""")


def test_tools():
    """Test individual aviation tools without the LLM."""
    print("\n" + "="*60)
    print("  TESTING AVIATION TOOLS")
    print("="*60)

    from tools.aviation_tools import (
        get_airport_info,
        get_weather_briefing,
        get_notam_info,
        get_aviation_regulations,
    )

    tests = [
        ("get_airport_info", get_airport_info, "VCBI"),
        ("get_airport_info", get_airport_info, "RML"),
        ("get_weather_briefing", get_weather_briefing, "CMB"),
        ("get_notam_info", get_notam_info, "fir colombo"),
        ("get_aviation_regulations", get_aviation_regulations, "ppl license"),
    ]

    for tool_name, tool_fn, arg in tests:
        print(f"\n📋 {tool_name}('{arg}')")
        print("-" * 40)
        try:
            result = tool_fn.invoke(arg)
            print(result)
        except Exception as e:
            print(f"ERROR: {e}")


def test_full_agent():
    """Test the full LangGraph agent with a sample query."""
    print("\n" + "="*60)
    print("  TESTING FULL AGENT PIPELINE")
    print("="*60)

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key == "your_openai_api_key_here":
        print("⚠️  Skipping full agent test — OPENAI_API_KEY not configured.")
        print("   Add your key to .env to test the full pipeline.")
        return

    from agent.graph import run_aviation_agent

    test_queries = [
        "What runways does Bandaranaike International Airport have?",
        "Give me a weather briefing for Colombo airport",
    ]

    for query in test_queries:
        print(f"\n🛫 Query: {query}")
        print("-" * 40)
        response = run_aviation_agent(query)
        print(response)
        print()


def show_memory_demo():
    """Demonstrate the conversation memory system."""
    print("\n" + "="*60)
    print("  CONVERSATION MEMORY DEMO")
    print("="*60)

    from memory.conversation_memory import ConversationMemory

    mem = ConversationMemory()
    session = mem.start_session("demo_session")

    # Simulate a conversation
    exchanges = [
        ("user", "What airports are in Sri Lanka?"),
        ("assistant", "Sri Lanka has 5 airports: VCBI (Colombo), VCCA (Hambantota)..."),
        ("user", "Tell me about the weather at VCBI"),
        ("assistant", "Current aviation weather at Colombo: Wind 220° at 12kts..."),
        ("user", "What PPL requirements does CAASL have?"),
    ]

    for role, content in exchanges:
        mem.add_message(role, content, session)

    summary = mem.get_session_summary(session)
    print(f"\nSession Summary:")
    print(f"  Total messages: {summary['total_messages']}")
    print(f"  User queries: {summary['user_queries']}")
    print(f"  Topics discussed: {summary['topics_discussed']}")

    print(f"\nContext string (last 3 messages):")
    print(mem.get_context_string(session, last_n=3))


if __name__ == "__main__":
    print_graph_structure()
    test_tools()
    show_memory_demo()
    test_full_agent()
    print("\n✅ Diagnostics complete. Run 'streamlit run app.py' to start the UI.\n")