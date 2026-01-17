"""Lightweight email helper supporting Outlook (pywin32) and Gmail fallback.

The module exposes :func:`send_email`, which accepts only three inputs:

``recipients``
    A sequence of email addresses (strings).  An explicit validation step
    ensures the list is non‑empty and that each address is well‑formed
    enough for mail clients that expect semicolon‑ or comma‑delimited
    strings.

``content``
    Flexible payload that can be a plain string, a mapping with rich
    metadata, or an :class:`EmailContent` instance.  HTML is the default
    format, but text bodies are also supported, and plain‑text fallbacks
    can be specified when sending HTML.

``open_before_sending``
    When ``True`` (the default) Outlook messages are displayed for user
    review rather than dispatched immediately.  Gmail cannot surface a UI
    in this environment, so preview mode returns the composed
    :class:`email.message.EmailMessage` without sending.

Outlook is used by default when ``pywin32`` is available.  Setting the
``ANALYSIS3054_EMAIL_PROVIDER`` environment variable to ``"gmail"`` forces
Gmail usage even when Outlook is present.  Gmail authentication relies on
``GMAIL_USER`` and ``GMAIL_PASSWORD`` environment variables (the latter is
expected to be an app password).
"""
from __future__ import annotations

import os
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage
from typing import Mapping, MutableMapping, Sequence

try:  # pywin32 is optional and only imported when available
    import win32com.client  # type: ignore
except ImportError:  # pragma: no cover - exercised indirectly via provider selection
    win32com = None  # type: ignore


@dataclass
class EmailContent:
    """Structured email content.

    Parameters
    ----------
    subject:
        Subject line for the email.
    body:
        Main body of the email.  When ``is_html`` is ``True`` this will be
        assigned to ``HTMLBody`` for Outlook and added as an HTML
        alternative for Gmail.
    is_html:
        Whether ``body`` should be treated as HTML.  Defaults to
        ``True`` to support rich content by default.
    plain_text:
        Optional plain‑text alternative.  When provided alongside HTML the
        plain text is used for clients or logging paths that prefer text.
    """

    subject: str
    body: str
    is_html: bool = True
    plain_text: str | None = None


def _normalize_recipients(recipients: Sequence[str]) -> list[str]:
    if not recipients:
        raise ValueError("At least one recipient is required")
    normalized = []
    for address in recipients:
        if not isinstance(address, str):
            raise TypeError("Recipient entries must be strings")
        trimmed = address.strip()
        if not trimmed:
            raise ValueError("Recipient entries cannot be empty strings")
        normalized.append(trimmed)
    return normalized


def _coerce_content(content: EmailContent | Mapping[str, str] | str) -> EmailContent:
    if isinstance(content, EmailContent):
        return content

    if isinstance(content, Mapping):
        mutable: MutableMapping[str, str] = dict(content)
        body = mutable.get("body", "")
        subject = mutable.get("subject", "Automated message")
        is_html = mutable.get("is_html", True)  # type: ignore[assignment]
        plain_text = mutable.get("plain_text")
        return EmailContent(subject=subject, body=body, is_html=bool(is_html), plain_text=plain_text)

    body_text = str(content)
    return EmailContent(subject="Automated message", body=body_text, is_html=True)


def _build_email_message(recipients: Sequence[str], content: EmailContent, sender: str | None) -> EmailMessage:
    message = EmailMessage()
    if sender:
        message["From"] = sender
    message["To"] = ", ".join(recipients)
    message["Subject"] = content.subject

    if content.is_html:
        message.set_content(content.plain_text or content.body, subtype="plain")
        message.add_alternative(content.body, subtype="html")
    else:
        message.set_content(content.body, subtype="plain")
    return message


def _outlook_available() -> bool:
    return win32com is not None


def _send_via_outlook(recipients: Sequence[str], content: EmailContent, open_before_sending: bool) -> object:
    outlook = win32com.client.Dispatch("Outlook.Application")  # type: ignore[attr-defined]
    mail = outlook.CreateItem(0)
    mail.To = "; ".join(recipients)
    mail.Subject = content.subject
    if content.is_html:
        mail.HTMLBody = content.body
        if content.plain_text:
            mail.Body = content.plain_text
    else:
        mail.Body = content.body

    if open_before_sending:
        mail.Display()
    else:
        mail.Send()
    return mail


def _send_via_gmail(recipients: Sequence[str], content: EmailContent, open_before_sending: bool) -> EmailMessage:
    username = os.getenv("GMAIL_USER")
    password = os.getenv("GMAIL_PASSWORD")
    if not username or not password:
        raise EnvironmentError("GMAIL_USER and GMAIL_PASSWORD environment variables are required for Gmail sending")

    message = _build_email_message(recipients, content, sender=username)

    if open_before_sending:
        # Preview mode: return the composed message without sending.
        return message

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(username, password)
        smtp.send_message(message)
    return message


def send_email(
    recipients: Sequence[str],
    content: EmailContent | Mapping[str, str] | str,
    open_before_sending: bool = True,
) -> object:
    """Send an email using Outlook (default) or Gmail fallback.

    Parameters
    ----------
    recipients:
        Sequence of recipient email addresses.  Addresses are normalized
        by stripping whitespace and validated to be non‑empty strings.
    content:
        Email content that can be a string (treated as HTML), a mapping
        with keys ``subject``, ``body``, ``is_html``, and ``plain_text``,
        or an :class:`EmailContent` instance for full control.
    open_before_sending:
        When ``True`` (default) Outlook opens the email window for manual
        review.  Gmail returns the composed :class:`EmailMessage` instead
        of sending when preview mode is requested.

    Returns
    -------
    The native mail object used for sending.  Outlook returns the COM
    mail item, while Gmail returns the composed :class:`EmailMessage`.
    """

    normalized_recipients = _normalize_recipients(recipients)
    normalized_content = _coerce_content(content)

    provider = os.getenv("ANALYSIS3054_EMAIL_PROVIDER", "outlook").lower()
    use_outlook = provider != "gmail" and _outlook_available()

    if use_outlook:
        return _send_via_outlook(normalized_recipients, normalized_content, open_before_sending)

    return _send_via_gmail(normalized_recipients, normalized_content, open_before_sending)
