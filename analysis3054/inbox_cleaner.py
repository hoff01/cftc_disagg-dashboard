"""Inbox cleanup utilities for IMAP mailboxes.

This module provides an IMAP-based inbox cleaner that can:

* Scan an entire mailbox for keyword matches across headers/body.
* Skip messages newer than a minimum age (default 30 days).
* Archive matching emails to a local filesystem path you control.
* Optionally copy/move matches to an IMAP archive mailbox.
* Delete (expunge) matches after successful archival.
* Run in dry-run mode to preview actions without modifying the mailbox.

The primary entry points are :class:`InboxCleaner` and the convenience
function :func:`clean_inbox`.

Example
-------
```python
from analysis3054.inbox_cleaner import (
    ArchiveConfig,
    ConnectionConfig,
    InboxCleaner,
    KeywordRule,
    RetentionPolicy,
)

cleaner = InboxCleaner(
    ConnectionConfig(
        host="imap.gmail.com",
        username="me@example.com",
        password="app-password",
    )
)

report = cleaner.clean(
    rules=[
        KeywordRule(keywords=["unsubscribe", "receipt"], fields={"subject", "body"}),
        KeywordRule(keywords=["invoice"], fields={"subject", "body", "from"}),
    ],
    retention=RetentionPolicy(min_age_days=30),
    archive=ArchiveConfig(path="/data/mail-archive", include_metadata=True),
    delete_matches=True,
    imap_archive_mailbox="Archive/Cleaned",
    dry_run=False,
)

print(report.summary)
```
"""
from __future__ import annotations

import email
import email.utils
import imaplib
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from email.message import Message
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ConnectionConfig:
    """IMAP connection settings."""

    host: str
    username: str
    password: str
    port: int = 993
    use_ssl: bool = True


@dataclass(frozen=True)
class KeywordRule:
    """Rule describing how to match messages against keywords.

    Parameters
    ----------
    keywords:
        List of keywords to search for.
    fields:
        Message fields to search. Allowed values are ``"subject"``,
        ``"body"``, ``"from"``, ``"to"``, and ``"cc"``.
    match_any:
        When ``True`` (default), any keyword can match. When ``False``,
        all keywords must match.
    case_sensitive:
        Whether the search is case sensitive.
    use_regex:
        When ``True``, keywords are treated as regular expressions.
    """

    keywords: Sequence[str]
    fields: set[str] = field(default_factory=lambda: {"subject", "body"})
    match_any: bool = True
    case_sensitive: bool = False
    use_regex: bool = False


@dataclass(frozen=True)
class RetentionPolicy:
    """Age-based retention policy."""

    min_age_days: int = 30

    def min_age_delta(self) -> timedelta:
        return timedelta(days=max(self.min_age_days, 0))


@dataclass(frozen=True)
class ArchiveConfig:
    """Local archival settings."""

    path: str | Path
    include_metadata: bool = False
    create_date_subdirs: bool = True

    def base_path(self) -> Path:
        return Path(self.path).expanduser().resolve()


@dataclass
class InboxCleanReport:
    """Summary of cleanup actions."""

    scanned: int = 0
    matched: int = 0
    archived: int = 0
    deleted: int = 0
    skipped_too_new: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        return (
            f"Scanned={self.scanned}, Matched={self.matched}, "
            f"Archived={self.archived}, Deleted={self.deleted}, "
            f"SkippedTooNew={self.skipped_too_new}, Errors={len(self.errors)}"
        )


class InboxCleaner:
    """IMAP inbox cleaner with keyword matching and archival."""

    def __init__(self, connection: ConnectionConfig) -> None:
        self.connection = connection

    def clean(
        self,
        rules: Sequence[KeywordRule],
        retention: RetentionPolicy | None = None,
        archive: ArchiveConfig | None = None,
        delete_matches: bool = True,
        imap_archive_mailbox: str | None = None,
        mailbox: str = "INBOX",
        dry_run: bool = True,
    ) -> InboxCleanReport:
        """Clean a mailbox according to the provided rules.

        Parameters
        ----------
        rules:
            Keyword rules that must be satisfied for a message to match.
        retention:
            Minimum age policy. Messages newer than ``min_age_days`` are
            skipped.
        archive:
            Local filesystem archive settings. When omitted, no local
            archive is written.
        delete_matches:
            Whether matching messages should be deleted (expunged).
        imap_archive_mailbox:
            IMAP mailbox name to copy/move matching messages into.
        mailbox:
            Mailbox to scan, default is ``INBOX``.
        dry_run:
            When ``True`` (default), no changes are made. Set to ``False``
            to perform archival/deletion actions.
        """

        if not rules:
            raise ValueError("At least one keyword rule is required")

        retention_policy = retention or RetentionPolicy()
        report = InboxCleanReport()
        min_age = retention_policy.min_age_delta()
        archive_root = archive.base_path() if archive else None

        with self._connect() as imap:
            imap.select(mailbox)
            status, message_ids = imap.search(None, "ALL")
            if status != "OK":
                raise RuntimeError("Failed to search mailbox")

            ids = message_ids[0].split()
            for msg_id in ids:
                report.scanned += 1
                try:
                    fetch_status, data = imap.fetch(msg_id, "(RFC822 INTERNALDATE)")
                    if fetch_status != "OK" or not data:
                        report.errors.append(f"Failed to fetch message {msg_id!r}")
                        continue

                    raw_message, internal_date = self._extract_message_parts(data)
                    message = email.message_from_bytes(raw_message)
                    message_date = self._message_date(message, internal_date)
                    if self._is_too_new(message_date, min_age):
                        report.skipped_too_new += 1
                        continue

                    if not self._matches_rules(message, rules):
                        continue

                    report.matched += 1
                    if dry_run:
                        continue

                    if archive_root:
                        self._archive_message(archive_root, message, raw_message, message_date, msg_id, archive)
                        report.archived += 1

                    if imap_archive_mailbox:
                        copied = self._archive_imap(imap, msg_id, imap_archive_mailbox)
                        if not copied:
                            report.errors.append(
                                f"Failed to copy message {msg_id!r} to IMAP archive {imap_archive_mailbox!r}"
                            )
                            continue

                    if delete_matches:
                        imap.store(msg_id, "+FLAGS", "\\Deleted")
                        report.deleted += 1
                except Exception as exc:  # pragma: no cover - defensive
                    report.errors.append(f"Message {msg_id!r}: {exc}")

            if delete_matches and not dry_run:
                imap.expunge()

        return report

    def _connect(self) -> imaplib.IMAP4:
        if self.connection.use_ssl:
            imap: imaplib.IMAP4 = imaplib.IMAP4_SSL(self.connection.host, self.connection.port)
        else:
            imap = imaplib.IMAP4(self.connection.host, self.connection.port)
        imap.login(self.connection.username, self.connection.password)
        return imap

    @staticmethod
    def _extract_message_parts(data: list[tuple[bytes, bytes]]) -> tuple[bytes, datetime | None]:
        raw_message = b""
        internal_date: datetime | None = None
        for item in data:
            if not isinstance(item, tuple):
                continue
            raw_message = item[1]
            if item[0]:
                decoded = item[0].decode("utf-8", errors="ignore")
                match = re.search(r'INTERNALDATE "([^"]+)"', decoded)
                if match:
                    internal_date = email.utils.parsedate_to_datetime(match.group(1))
        return raw_message, internal_date

    @staticmethod
    def _message_date(message: Message, fallback: datetime | None) -> datetime:
        header = message.get("Date")
        if header:
            try:
                parsed = email.utils.parsedate_to_datetime(header)
                if parsed:
                    return parsed
            except (TypeError, ValueError):
                pass
        if fallback:
            return fallback
        return datetime.now(tz=UTC)

    @staticmethod
    def _is_too_new(message_date: datetime, min_age: timedelta) -> bool:
        now = datetime.now(tz=message_date.tzinfo or UTC)
        return now - message_date < min_age

    @staticmethod
    def _matches_rules(message: Message, rules: Sequence[KeywordRule]) -> bool:
        subject = message.get("Subject", "")
        from_ = message.get("From", "")
        to = message.get("To", "")
        cc = message.get("Cc", "")
        body = InboxCleaner._extract_body(message)

        for rule in rules:
            haystack = []
            if "subject" in rule.fields:
                haystack.append(subject)
            if "from" in rule.fields:
                haystack.append(from_)
            if "to" in rule.fields:
                haystack.append(to)
            if "cc" in rule.fields:
                haystack.append(cc)
            if "body" in rule.fields:
                haystack.append(body)

            if not haystack:
                continue

            match_result = InboxCleaner._match_keywords(haystack, rule)
            if match_result:
                return True
        return False

    @staticmethod
    def _match_keywords(haystack: Iterable[str], rule: KeywordRule) -> bool:
        tokens = [token for token in rule.keywords if token]
        if not tokens:
            return False

        contents = "\n".join(haystack)
        if not rule.case_sensitive:
            contents = contents.lower()
            tokens = [token.lower() for token in tokens]

        matches = []
        for token in tokens:
            if rule.use_regex:
                matches.append(bool(re.search(token, contents)))
            else:
                matches.append(token in contents)

        return any(matches) if rule.match_any else all(matches)

    @staticmethod
    def _extract_body(message: Message) -> str:
        if message.is_multipart():
            parts = [InboxCleaner._extract_body(part) for part in message.get_payload()]
            return "\n".join(part for part in parts if part)

        content_type = message.get_content_type()
        payload = message.get_payload(decode=True)
        if payload is None:
            return ""
        charset = message.get_content_charset() or "utf-8"
        text = payload.decode(charset, errors="replace")
        if content_type == "text/html":
            text = re.sub(r"<[^>]+>", " ", text)
        return text

    def _archive_message(
        self,
        root: Path,
        message: Message,
        raw_message: bytes,
        message_date: datetime,
        msg_id: bytes,
        archive: ArchiveConfig,
    ) -> None:
        date_part = message_date.astimezone(UTC).strftime("%Y-%m-%d")
        archive_dir = root / date_part if archive.create_date_subdirs else root
        archive_dir.mkdir(parents=True, exist_ok=True)

        subject = message.get("Subject", "(no subject)")
        safe_subject = re.sub(r"[^A-Za-z0-9._-]+", "_", subject).strip("_")
        filename = f"{date_part}_{safe_subject}_{msg_id.decode(errors='ignore')}.eml"
        archive_path = archive_dir / filename
        archive_path.write_bytes(raw_message)

        if archive.include_metadata:
            metadata = {
                "subject": subject,
                "from": message.get("From", ""),
                "to": message.get("To", ""),
                "cc": message.get("Cc", ""),
                "date": message_date.isoformat(),
                "message_id": message.get("Message-Id", ""),
            }
            metadata_path = archive_path.with_suffix(".json")
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @staticmethod
    def _archive_imap(imap: imaplib.IMAP4, msg_id: bytes, mailbox: str) -> bool:
        status, _ = imap.copy(msg_id, mailbox)
        return status == "OK"


def clean_inbox(
    connection: ConnectionConfig,
    rules: Sequence[KeywordRule],
    retention: RetentionPolicy | None = None,
    archive: ArchiveConfig | None = None,
    delete_matches: bool = True,
    imap_archive_mailbox: str | None = None,
    mailbox: str = "INBOX",
    dry_run: bool = True,
) -> InboxCleanReport:
    """Convenience wrapper for :class:`InboxCleaner`.

    Returns the :class:`InboxCleanReport` summarizing the run.
    """

    cleaner = InboxCleaner(connection)
    return cleaner.clean(
        rules=rules,
        retention=retention,
        archive=archive,
        delete_matches=delete_matches,
        imap_archive_mailbox=imap_archive_mailbox,
        mailbox=mailbox,
        dry_run=dry_run,
    )
