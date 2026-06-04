"""IMAP helpers and email parsing."""

from __future__ import annotations

import email
import imaplib
import re
from dataclasses import asdict, dataclass
from email.header import decode_header
from email.utils import parsedate_to_datetime
from typing import Iterable


@dataclass
class EmailMessage:
    uid: str
    subject: str
    sender: str
    date: str
    body: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> EmailMessage:
        return cls(**data)


def decode_mime_header(value: str | None) -> str:
    if not value:
        return ""
    parts = decode_header(value)
    out: list[str] = []
    for chunk, charset in parts:
        if isinstance(chunk, bytes):
            out.append(chunk.decode(charset or "utf-8", errors="replace"))
        else:
            out.append(chunk)
    return "".join(out)


def html_to_text(html: str) -> str:
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.I | re.S)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</p>", "\n\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def extract_body(msg: email.message.Message) -> str:
    plain_parts: list[str] = []
    html_parts: list[str] = []

    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_maintype() == "multipart":
                continue
            disposition = (part.get("Content-Disposition") or "").lower()
            if "attachment" in disposition:
                continue
            ctype = part.get_content_type()
            try:
                payload = part.get_payload(decode=True)
                if not payload:
                    continue
                charset = part.get_content_charset() or "utf-8"
                text = payload.decode(charset, errors="replace")
            except Exception:
                continue
            if ctype == "text/plain":
                plain_parts.append(text)
            elif ctype == "text/html":
                html_parts.append(text)
    else:
        try:
            payload = msg.get_payload(decode=True)
            if payload:
                charset = msg.get_content_charset() or "utf-8"
                text = payload.decode(charset, errors="replace")
                if msg.get_content_type() == "text/html":
                    html_parts.append(text)
                else:
                    plain_parts.append(text)
        except Exception:
            pass

    if plain_parts:
        return "\n\n".join(plain_parts).strip()
    if html_parts:
        return html_to_text("\n\n".join(html_parts))
    return ""


def truncate(text: str, max_chars: int = 6000) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20] + "\n\n[... truncated]"


def format_emails_for_llm(emails: Iterable[EmailMessage]) -> str:
    blocks: list[str] = []
    for i, em in enumerate(emails, 1):
        blocks.append(
            f"--- Email {i} ---\n"
            f"From: {em.sender}\n"
            f"Date: {em.date}\n"
            f"Subject: {em.subject}\n\n"
            f"{em.body or '(empty body)'}"
        )
    return "\n\n".join(blocks)


def fetch_emails_imap(
    *,
    address: str,
    password: str,
    imap_host: str,
    imap_port: int,
    mailbox: str = "INBOX",
    limit: int = 10,
    unread_only: bool = True,
) -> list[EmailMessage]:
    messages: list[EmailMessage] = []
    with imaplib.IMAP4_SSL(imap_host, imap_port) as imap:
        imap.login(address, password)
        imap.select(mailbox)
        criterion = "UNSEEN" if unread_only else "ALL"
        _, data = imap.search(None, criterion)
        ids = data[0].split()
        if not ids:
            return []
        for uid in reversed(ids[-limit:]):
            _, msg_data = imap.fetch(uid, "(RFC822)")
            if not msg_data or not msg_data[0]:
                continue
            raw = msg_data[0][1]
            if not isinstance(raw, bytes):
                continue
            msg = email.message_from_bytes(raw)
            subject = decode_mime_header(msg.get("Subject"))
            sender = decode_mime_header(msg.get("From"))
            date_hdr = msg.get("Date")
            try:
                date_str = (
                    parsedate_to_datetime(date_hdr).isoformat() if date_hdr else ""
                )
            except Exception:
                date_str = date_hdr or ""
            body = truncate(extract_body(msg))
            messages.append(
                EmailMessage(
                    uid=uid.decode() if isinstance(uid, bytes) else str(uid),
                    subject=subject or "(no subject)",
                    sender=sender,
                    date=date_str,
                    body=body,
                )
            )
    return messages
