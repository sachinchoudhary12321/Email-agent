"""LangGraph workflow: fetch inbox → extract insights → build brief."""

from __future__ import annotations

from typing import Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph

from email_utils import EmailMessage, fetch_emails_imap, format_emails_for_llm

EXTRACT_SYSTEM = """You are an email assistant. For each email, extract only what is useful:
- Main point (1–2 sentences)
- Action items (who should do what, if any)
- Deadlines or dates mentioned
- People / organizations worth noting
- Urgency (low / medium / high) with a short reason

Skip ads, newsletter fluff, and legal footers. If an email is spam or empty, say "Low value — skip" and one line why.

Respond in clear markdown with one section per email (use the subject as the heading)."""

BRIEF_SYSTEM = """You are an executive assistant. Given per-email notes, write a short inbox brief:
1. **Top priorities** — bullet list, most urgent first
2. **Action items** — combined checklist with sender/context
3. **Deadlines** — dates mentioned across emails
4. **Safe to ignore** — low-value threads in one line each

Keep it scannable. Do not repeat full email bodies."""


class EmailAgentState(TypedDict, total=False):
    limit: int
    unread_only: bool
    mailbox: str
    emails: list[dict]
    insights: str
    summary: str


class ImapConfig(TypedDict):
    address: str
    password: str
    imap_host: str
    imap_port: int


def build_email_graph(
    imap_config: ImapConfig,
    llm: ChatGoogleGenerativeAI,
):
    def fetch_inbox(state: EmailAgentState) -> dict:
        emails = fetch_emails_imap(
            address=imap_config["address"],
            password=imap_config["password"],
            imap_host=imap_config["imap_host"],
            imap_port=imap_config["imap_port"],
            mailbox=state.get("mailbox", "INBOX"),
            limit=state.get("limit", 10),
            unread_only=state.get("unread_only", True),
        )
        return {"emails": [e.to_dict() for e in emails]}

    def handle_empty(state: EmailAgentState) -> dict:
        return {"summary": "No emails matched your filters."}

    def extract_insights(state: EmailAgentState) -> dict:
        emails = [EmailMessage.from_dict(d) for d in state["emails"]]
        content = format_emails_for_llm(emails)
        response = llm.invoke(
            [
                SystemMessage(content=EXTRACT_SYSTEM),
                HumanMessage(
                    content=(
                        f"Extract useful content from these {len(emails)} email(s):\n\n"
                        f"{content}"
                    )
                ),
            ]
        )
        text = response.content
        insights = text if isinstance(text, str) else str(text)
        return {"insights": insights.strip()}

    def build_brief(state: EmailAgentState) -> dict:
        response = llm.invoke(
            [
                SystemMessage(content=BRIEF_SYSTEM),
                HumanMessage(
                    content=(
                        "Create an inbox brief from these per-email notes:\n\n"
                        f"{state['insights']}"
                    )
                ),
            ]
        )
        text = response.content
        summary = text if isinstance(text, str) else str(text)
        return {"summary": summary.strip()}

    def route_after_fetch(state: EmailAgentState) -> Literal["extract_insights", "handle_empty"]:
        if not state.get("emails"):
            return "handle_empty"
        return "extract_insights"

    workflow = StateGraph(EmailAgentState)
    workflow.add_node("fetch_inbox", fetch_inbox)
    workflow.add_node("extract_insights", extract_insights)
    workflow.add_node("build_brief", build_brief)
    workflow.add_node("handle_empty", handle_empty)

    workflow.add_edge(START, "fetch_inbox")
    workflow.add_conditional_edges(
        "fetch_inbox",
        route_after_fetch,
        {"extract_insights": "extract_insights", "handle_empty": "handle_empty"},
    )
    workflow.add_edge("extract_insights", "build_brief")
    workflow.add_edge("build_brief", END)
    workflow.add_edge("handle_empty", END)

    return workflow.compile()
