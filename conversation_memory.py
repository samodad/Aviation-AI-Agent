"""
Conversation Memory for SkyMind SL
Session-based memory manager for the aviation agent.
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# Simple In-Memory Store (no API key needed)
# ──────────────────────────────────────────────

class ConversationMemory:
    """
    Manages conversation history and session memory for the aviation agent.
    Stores conversations with timestamps and provides retrieval.
    """

    def __init__(self, max_history: int = 20):
        self.max_history = max_history
        self.sessions: Dict[str, List[Dict]] = {}
        self.current_session_id: Optional[str] = None

    def start_session(self, session_id: str = None) -> str:
        """Start a new conversation session."""
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.current_session_id = session_id
        if session_id not in self.sessions:
            self.sessions[session_id] = []

        return session_id

    def add_message(self, role: str, content: str, session_id: str = None):
        """
        Add a message to the conversation history.
        
        Args:
            role: 'user' or 'assistant'
            content: Message content
            session_id: Session identifier (uses current if None)
        """
        sid = session_id or self.current_session_id
        if sid is None:
            sid = self.start_session()

        if sid not in self.sessions:
            self.sessions[sid] = []

        self.sessions[sid].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })

        # Keep only recent messages
        if len(self.sessions[sid]) > self.max_history:
            self.sessions[sid] = self.sessions[sid][-self.max_history:]

    def get_history(self, session_id: str = None, last_n: int = 10) -> List[Dict]:
        """
        Retrieve conversation history for a session.
        
        Args:
            session_id: Session to retrieve (uses current if None)
            last_n: Number of recent messages to return
            
        Returns:
            List of message dicts with role, content, timestamp
        """
        sid = session_id or self.current_session_id
        if sid is None or sid not in self.sessions:
            return []

        return self.sessions[sid][-last_n:]

    def get_context_string(self, session_id: str = None, last_n: int = 6) -> str:
        """
        Get conversation history as a formatted string for context injection.
        """
        history = self.get_history(session_id, last_n)
        if not history:
            return ""

        lines = ["Previous conversation context:"]
        for msg in history:
            role_label = "User" if msg["role"] == "user" else "SkyMind"
            lines.append(f"  {role_label}: {msg['content'][:200]}...")

        return "\n".join(lines)

    def clear_session(self, session_id: str = None):
        """Clear a session's history."""
        sid = session_id or self.current_session_id
        if sid and sid in self.sessions:
            self.sessions[sid] = []

    def get_session_summary(self, session_id: str = None) -> Dict:
        """Get a summary of session statistics."""
        sid = session_id or self.current_session_id
        if not sid or sid not in self.sessions:
            return {"messages": 0, "session_id": sid}

        history = self.sessions[sid]
        user_msgs = [m for m in history if m["role"] == "user"]
        topics = self._extract_aviation_topics(history)

        return {
            "session_id": sid,
            "total_messages": len(history),
            "user_queries": len(user_msgs),
            "topics_discussed": topics,
            "started": history[0]["timestamp"] if history else None,
            "last_activity": history[-1]["timestamp"] if history else None,
        }

    def _extract_aviation_topics(self, history: List[Dict]) -> List[str]:
        """Extract key aviation topics from conversation history."""
        topics = set()
        keywords = {
            "weather": ["weather", "metar", "wind", "visibility", "cloud"],
            "airports": ["airport", "vcbi", "vcca", "vccb", "cmb", "rml", "runway"],
            "regulations": ["cpl", "ppl", "atpl", "license", "regulation", "caasl"],
            "notam": ["notam", "restriction", "airspace", "fir"],
            "flights": ["flight", "airline", "srilankan", "departure", "arrival"],
        }

        full_text = " ".join(
            m["content"].lower() for m in history if m["role"] == "user"
        )

        for topic, kws in keywords.items():
            if any(kw in full_text for kw in kws):
                topics.add(topic)

        return list(topics)

    def export_session(self, session_id: str = None) -> str:
        """Export session as JSON string."""
        sid = session_id or self.current_session_id
        data = {
            "session_id": sid,
            "exported_at": datetime.now().isoformat(),
            "messages": self.sessions.get(sid, []),
        }
        return json.dumps(data, indent=2)


# ──────────────────────────────────────────────
# Global Memory Instance
# ──────────────────────────────────────────────

# Singleton memory instance shared across the app
_memory_instance = None

def get_memory() -> ConversationMemory:
    """Get or create the global memory instance."""
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = ConversationMemory(max_history=30)
    return _memory_instance


def reset_memory():
    """Reset the global memory (useful for testing)."""
    global _memory_instance
    _memory_instance = ConversationMemory(max_history=30)
    return _memory_instance