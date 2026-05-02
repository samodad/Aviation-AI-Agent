# ✈️ SkyMind SL — Sri Lanka Aviation AI Agent

> **An autonomous multi-agent AI system for Sri Lanka aviation, built with LangGraph, LangChain, and Streamlit.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.1+-green.svg)](https://github.com/langchain-ai/langgraph)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Overview

**SkyMind SL** is an autonomous AI agent designed for the Sri Lanka aviation domain. It uses **LangGraph** for multi-agent orchestration, **LangChain** for tool integration, and **Streamlit** for an interactive aviation-themed dashboard.

The agent can:
- 🏢 Retrieve detailed information on all Sri Lankan airports (VCBI, VCCA, VCCB, VCCT, VCCJ)
- 🌤️ Generate aviation weather briefings (live via OpenWeatherMap or demo mode)
- 📋 Provide NOTAM guidance and Colombo FIR information
- 📚 Answer questions on CAASL regulations (PPL, CPL, ATPL licensing, FPL filing)
- 🔍 Search the web for current aviation news (via Tavily)
- 💬 Maintain conversation memory across the session

---

## 🏗️ Architecture

```
User Query (Streamlit)
        ↓
   LangGraph Orchestrator
        ↓
  ┌─────────────────────────────────────┐
  │         Agent Node (GPT-4o-mini)    │
  │   Reason → Decide → Act (ReAct)     │
  └──────────┬──────────────────────────┘
             │ Tool Calls
             ↓
  ┌──────────────────────────────────────┐
  │         Tool Node (LangChain)        │
  │  • get_airport_info                  │
  │  • get_weather_briefing              │
  │  • get_notam_info                    │
  │  • get_aviation_regulations          │
  │  • search_aviation_web (Tavily)      │
  └──────────┬───────────────────────────┘
             │ Tool Results
             ↓
  ┌──────────────────────────────────────┐
  │       Synthesizer Node               │
  │   Format → Markdown Response         │
  └──────────┬───────────────────────────┘
             ↓
   Final Response (Streamlit Chat)
```

### Key Technologies
| Component | Technology |
|---|---|
| Agent Orchestration | **LangGraph** (State Machine) |
| LLM | **GPT-4o-mini** via OpenAI API |
| Tool Framework | **LangChain** Tools |
| Web Search | **Tavily** API |
| Weather Data | **OpenWeatherMap** API |
| Memory | Custom session-based conversation memory |
| Frontend | **Streamlit** (aviation dark theme) |

---

## 📁 Project Structure

```
skymind_sl/
├── app.py                        # Streamlit frontend (main entry point)
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variables template
├── README.md                     # This file
│
├── agent/
│   ├── __init__.py
│   └── graph.py                  # LangGraph state machine & agent nodes
│
├── tools/
│   ├── __init__.py
│   └── aviation_tools.py         # LangChain tools (airport, weather, NOTAM, etc.)
│
└── memory/
    ├── __init__.py
    └── conversation_memory.py    # Session-based conversation memory
```

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.10 or higher
- OpenAI API key (required)
- Tavily API key (optional, for web search)
- OpenWeatherMap API key (optional, for live weather)

### Step 1: Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/skymind-sl.git
cd skymind-sl
```

### Step 2: Create a Virtual Environment
```bash
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure API Keys
```bash
# Copy the example env file
cp env.example .env

# Edit .env and add your API keys
# (Use any text editor)
```

Your `.env` file should look like:
```env
OPENAI_API_KEY=sk-your-key-here
TAVILY_API_KEY=tvly-your-key-here        # Optional
OPENWEATHER_API_KEY=your-key-here        # Optional
```

**Where to get API keys:**
- **OpenAI**: https://platform.openai.com/api-keys
- **Tavily** (free tier available): https://tavily.com
- **OpenWeatherMap** (free tier available): https://openweathermap.org/api

### Step 5: Run the Application
```bash
streamlit run app.py
```

Open your browser to: **(https://aviation-ai-agent-hvklqfksaunmrdj2jmuaqb.streamlit.app/)**
Check out in: **https://youtube.com/shorts/xNRcwjaXxwU**


---

## 💡 Usage Examples

Once running, try these queries:

| Query | What it demonstrates |
|---|---|
| *"Tell me about VCBI airport"* | Airport info tool |
| *"Give me a weather briefing for Colombo"* | Weather tool + aviation formatting |
| *"What are the NOTAM procedures for Colombo FIR?"* | NOTAM tool |
| *"How do I get a CPL license in Sri Lanka?"* | Regulations tool |
| *"What airlines fly from CMB to Singapore?"* | Web search tool (Tavily) |
| *"What is the QNH for Ratmalana right now?"* | Weather tool (VCCB) |
| *"Explain VFR vs IFR flying in Sri Lanka"* | General aviation Q&A |

---

## 🔑 Features

### 1. LangGraph Multi-Agent Orchestration
- **State Machine** architecture with typed state (`AgentState`)
- **ReAct pattern**: Reason, Act, Observe loop
- **Conditional edges**: Dynamically routes to tools or synthesis
- **Loop guard**: Prevents infinite reasoning cycles

### 2. Specialized Aviation Tools
- **Airport Info**: Complete data on all 5 Sri Lankan airports
- **Weather Briefing**: Live data via OpenWeatherMap, demo mode without key
- **NOTAM Guidance**: Colombo FIR, restricted airspace, drone regulations
- **Regulations**: CAASL licensing, medical requirements, FPL procedures
- **Web Search**: Real-time search via Tavily API

### 3. Conversation Memory
- Session-based memory with configurable history length
- Topic extraction from conversation history
- Session export functionality

### 4. Aviation-Themed UI
- Dark cockpit-inspired interface
- Real-time status metrics
- Quick-access airport buttons
- One-click preset queries for common aviation topics

---

## 🛫 Sri Lanka Aviation Domain Knowledge

The agent has built-in knowledge about:

| Airport | ICAO | IATA | Location |
|---|---|---|---|
| Bandaranaike International | VCBI | CMB | Colombo (Katunayake) |
| Mattala Rajapaksa International | VCCA | HRI | Hambantota |
| Ratmalana Airport | VCCB | RML | Colombo (Ratmalana) |
| China Bay Airport | VCCT | TRR | Trincomalee |
| Jaffna International Airport | VCCJ | JAF | Jaffna (Palaly) |

**Key Authorities:**
- **CAASL** — Civil Aviation Authority of Sri Lanka (caasl.gov.lk)
- **AASL** — Airport & Aviation Services Ltd (airport.lk)
- **Colombo FIR** — VCCF (Colombo CONTROL, 128.9 MHz)

---

## ⚠️ Disclaimer

**SkyMind SL is an educational AI assistant.** It is NOT a certified aviation information system. Always verify:
- Weather data with official AASL Meteorological services
- NOTAMs via official AIS/AASL channels
- Regulations with CAASL directly
- ATC clearances via licensed Air Traffic Control

**Do not use this system for actual flight operations.**

---

## 📖 Learning Outcomes Demonstrated

| Outcome | Implementation |
|---|---|
| Autonomous AI Agent Design | LangGraph state machine with conditional routing |
| Planning & Reasoning | ReAct (Reason + Act) pattern in agent node |
| Memory Integration | Session-based ConversationMemory class |
| Tool Use | 5 specialized LangChain tools |
| Real-World Domain | Sri Lanka aviation (CAASL, AASL, airports) |
| Interactive UI | Streamlit with aviation-themed design |
| Modular Code | Separated agent/, tools/, memory/ packages |

---

## 👨‍💻 Author

**Samoda De Silva**  
*Data Science Student*  
*AI Agent Development — Aviation Domain*

---

## 📄 License

MIT License — See [LICENSE](LICENSE) for details.
