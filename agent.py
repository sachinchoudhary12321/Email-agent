"""Email agent powered by a LangGraph workflow."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from email_utils import EmailMessage
from graph import ImapConfig, build_email_graph

load_dotenv()


class EmailAgent:
    """Runs fetch → extract → brief via LangGraph."""

    def __init__(
        self,
        address: str | None = None,
        password: str | None = None,
        imap_host: str | None = None,
        imap_port: int | None = None,
        gemini_api_key: str | None = None,
        model: str | None = None,
    ):
        self.address = address or os.environ["EMAIL_ADDRESS"]
        self.password = password or os.environ["EMAIL_PASSWORD"]
        self.imap_host = imap_host or os.getenv("IMAP_HOST", "imap.gmail.com")
        self.imap_port = imap_port or int(os.getenv("IMAP_PORT", "993"))
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for summarization.")

        llm = ChatGoogleGenerativeAI(
            model=self.model,
            google_api_key=api_key,
            temperature=0.3,
        )
        imap_config: ImapConfig = {
            "address": self.address,
            "password": self.password,
            "imap_host": self.imap_host,
            "imap_port": self.imap_port,
        }
        self._graph = build_email_graph(imap_config, llm)

    def run(
        self,
        *,
        limit: int = 10,
        unread_only: bool = True,
        mailbox: str = "INBOX",
    ) -> tuple[list[EmailMessage], str, str | None]:
        """Returns (emails, summary, insights). insights is None when inbox was empty."""
        result = self._graph.invoke(
            {
                "limit": limit,
                "unread_only": unread_only,
                "mailbox": mailbox,
            }
        )
        emails = [EmailMessage.from_dict(d) for d in result.get("emails", [])]
        summary = result.get("summary", "")
        insights = result.get("insights")
        return emails, summary, insights
