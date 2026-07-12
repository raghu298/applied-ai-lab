"""
The message bus: two SQLite files per session, one writer each.

This is Chapter 2 of the handbook made real. The host and the agent worker
never share a socket -- they talk through two SQLite files:

    inbound.db   written ONLY by the host      (your messages in)
    outbound.db  written ONLY by the worker     (the agent's replies out)

Two rules make it robust:
  * even/odd sequencing -- the host allocates only EVEN sequence numbers, the
    worker only ODD ones. Collision-free IDs with zero coordination, and the
    parity of any row tells you who wrote it.
  * process-after-write -- the worker marks an inbound row "done" only AFTER
    its reply is safely in outbound.db. Crash in between? The row is still
    pending, so the message is retried, not lost.
"""

import os
import sqlite3
from dataclasses import dataclass


@dataclass
class Row:
    seq: int
    text: str


def _connect(path: str) -> sqlite3.Connection:
    con = sqlite3.connect(path, timeout=5)
    con.execute("PRAGMA journal_mode=WAL")      # safe concurrent read/write
    con.execute("PRAGMA synchronous=NORMAL")
    return con


class SessionBus:
    def __init__(self, session_dir: str):
        os.makedirs(session_dir, exist_ok=True)
        self.inbound = _connect(os.path.join(session_dir, "inbound.db"))
        self.outbound = _connect(os.path.join(session_dir, "outbound.db"))
        self.inbound.execute(
            "CREATE TABLE IF NOT EXISTS msgs("
            "seq INTEGER PRIMARY KEY, text TEXT, done INT DEFAULT 0)")
        self.outbound.execute(
            "CREATE TABLE IF NOT EXISTS msgs("
            "seq INTEGER PRIMARY KEY, text TEXT, delivered INT DEFAULT 0)")
        self.inbound.commit()
        self.outbound.commit()

    # --- host side: writes EVEN seqs to inbound, reads outbound ------------
    def put_inbound(self, text: str) -> int:
        cur = self.inbound.execute("SELECT MAX(seq) FROM msgs").fetchone()[0]
        seq = (cur if cur is not None else 0) + 2      # 2, 4, 6, ... (even)
        seq += seq % 2                                  # force even if cur was odd-seeded
        self.inbound.execute("INSERT INTO msgs(seq, text) VALUES(?, ?)", (seq, text))
        self.inbound.commit()
        return seq

    def pending_outbound(self) -> list[Row]:
        rows = self.outbound.execute(
            "SELECT seq, text FROM msgs WHERE delivered=0 ORDER BY seq").fetchall()
        return [Row(*r) for r in rows]

    def mark_delivered(self, seq: int) -> None:
        self.outbound.execute("UPDATE msgs SET delivered=1 WHERE seq=?", (seq,))
        self.outbound.commit()

    # --- worker side: reads inbound, writes ODD seqs to outbound ------------
    def pending_inbound(self) -> list[Row]:
        rows = self.inbound.execute(
            "SELECT seq, text FROM msgs WHERE done=0 ORDER BY seq").fetchall()
        return [Row(*r) for r in rows]

    def put_outbound(self, text: str) -> int:
        cur = self.outbound.execute("SELECT MAX(seq) FROM msgs").fetchone()[0]
        seq = (cur if cur is not None else -1) + 2      # 1, 3, 5, ... (odd)
        if seq % 2 == 0:
            seq += 1
        self.outbound.execute("INSERT INTO msgs(seq, text) VALUES(?, ?)", (seq, text))
        self.outbound.commit()
        return seq

    def mark_done(self, seq: int) -> None:
        self.inbound.execute("UPDATE msgs SET done=1 WHERE seq=?", (seq,))
        self.inbound.commit()
