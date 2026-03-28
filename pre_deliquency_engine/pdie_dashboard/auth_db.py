"""
auth_db.py — PDIE Authentication & Analyst Queue Database
==========================================================
Handles:
  - User registration / login (bcrypt-hashed passwords)
  - SQLite schema: users + analyst_assignments
  - Analyst queue CRUD operations
  - Seed dummy data on first run
"""

import sqlite3
import hashlib
import os
import re
from pathlib import Path
from datetime import datetime

# ── DB location ────────────────────────────────────────────────────
DB_PATH = Path(__file__).parent / "pdie_auth.db"

# ── Bcrypt-compatible password hashing via hashlib (no extra deps) ──
# We use PBKDF2-HMAC-SHA256 (stdlib, zero deps, strong enough for demo)
_ITERATIONS = 260_000
_SALT_LEN   = 32


def _hash_password(password: str) -> str:
    salt = os.urandom(_SALT_LEN)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _ITERATIONS)
    return salt.hex() + ":" + dk.hex()


def _verify_password(password: str, stored: str) -> bool:
    try:
        salt_hex, dk_hex = stored.split(":")
        salt = bytes.fromhex(salt_hex)
        dk   = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _ITERATIONS)
        return dk.hex() == dk_hex
    except Exception:
        return False


# ── Schema ─────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name     TEXT    NOT NULL,
    email         TEXT    NOT NULL UNIQUE,
    password_hash TEXT    NOT NULL,
    role          TEXT    NOT NULL DEFAULT 'Analyst',
    created_at    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS analyst_assignments (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id   TEXT    NOT NULL,
    customer_name TEXT,
    risk_score    REAL,
    risk_category TEXT,
    analyst_email TEXT    NOT NULL,
    analyst_name  TEXT    NOT NULL,
    assigned_at   TEXT    NOT NULL,
    status        TEXT    NOT NULL DEFAULT 'Active',
    notes         TEXT    DEFAULT '',
    FOREIGN KEY (analyst_email) REFERENCES users(email)
);
"""

_DUMMY_USERS = [
    ("Aadi Jain",      "aadi@barclays.com",    "Password@123", "Admin"),
    ("Priya Sharma",   "priya@barclays.com",   "Password@123", "Analyst"),
    ("Rahul Mehta",    "rahul@barclays.com",   "Password@123", "Analyst"),
    ("Sneha Patel",    "sneha@barclays.com",   "Password@123", "Analyst"),
    ("Vikram Singh",   "vikram@barclays.com",  "Password@123", "Analyst"),
]

_DUMMY_CUSTOMERS = [
    ("C-1001", "Amit Kumar",    91.2, "CRITICAL"),
    ("C-1002", "Deepa Nair",    85.7, "CRITICAL"),
    ("C-1003", "Ravi Reddy",    78.4, "HIGH"),
    ("C-1004", "Sunita Iyer",   74.1, "HIGH"),
    ("C-1005", "Manoj Tiwari",  71.8, "HIGH"),
    ("C-1006", "Kavya Menon",   67.3, "MEDIUM"),
    ("C-1007", "Ajay Verma",    63.9, "MEDIUM"),
    ("C-1008", "Pooja Shah",    61.2, "MEDIUM"),
    ("C-1009", "Rohit Yadav",   55.4, "MEDIUM"),
    ("C-1010", "Ananya Das",    52.0, "MEDIUM"),
    ("C-1011", "Karthik Bose",  88.3, "CRITICAL"),
    ("C-1012", "Meera Pillai",  83.5, "CRITICAL"),
    ("C-1013", "Suresh Joshi",  76.2, "HIGH"),
    ("C-1014", "Neha Gupta",    70.8, "HIGH"),
    ("C-1015", "Arjun Nair",    65.1, "MEDIUM"),
]

# Analyst assignment distribution (analyst_email -> list of customer indices)
_ASSIGNMENTS = {
    "priya@barclays.com":  [0, 1, 5, 6],
    "rahul@barclays.com":  [2, 3, 10, 11],
    "sneha@barclays.com":  [4, 7, 12],
    "vikram@barclays.com": [8, 9, 13, 14],
}


# ── Connection helper ───────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # WAL mode: allows concurrent readers even while a write is in progress.
    # This ensures admin session B can read data written by analyst session A immediately.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db():
    """Create tables and seed dummy data if not already present."""
    conn = _get_conn()
    conn.executescript(_DDL)
    conn.commit()

    # Seed users
    for full_name, email, password, role in _DUMMY_USERS:
        existing = conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
        if not existing:
            conn.execute(
                "INSERT INTO users (full_name, email, password_hash, role, created_at) VALUES (?,?,?,?,?)",
                (full_name, email, _hash_password(password), role, datetime.now().isoformat())
            )

    conn.commit()

    # Seed analyst assignments
    existing_assignments = conn.execute("SELECT COUNT(*) as c FROM analyst_assignments").fetchone()["c"]
    if existing_assignments == 0:
        # Get analyst names from users table
        analyst_names = {
            row["email"]: row["full_name"]
            for row in conn.execute("SELECT email, full_name FROM users WHERE role = 'Analyst'").fetchall()
        }
        for analyst_email, cust_indices in _ASSIGNMENTS.items():
            analyst_name = analyst_names.get(analyst_email, analyst_email)
            for idx in cust_indices:
                cid, cname, score, cat = _DUMMY_CUSTOMERS[idx]
                conn.execute(
                    "INSERT INTO analyst_assignments "
                    "(customer_id, customer_name, risk_score, risk_category, analyst_email, analyst_name, assigned_at, status) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (cid, cname, score, cat, analyst_email, analyst_name, datetime.now().isoformat(), "Active")
                )
        conn.commit()

    conn.close()


# ── Auth API ────────────────────────────────────────────────────────

def register_user(full_name: str, email: str, password: str, role: str) -> dict:
    """Register a new user. Returns {'ok': True} or {'ok': False, 'error': '...'}."""
    if not re.match(r"^[^@]+@[^@]+\.[^@]+$", email):
        return {"ok": False, "error": "Invalid email format."}
    if len(password) < 8:
        return {"ok": False, "error": "Password must be at least 8 characters."}
    if not re.search(r"[A-Z]", password):
        return {"ok": False, "error": "Password must contain at least one uppercase letter."}
    if not re.search(r"[0-9]", password):
        return {"ok": False, "error": "Password must contain at least one digit."}
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return {"ok": False, "error": "Password must contain at least one special character."}

    conn = _get_conn()
    existing = conn.execute("SELECT id FROM users WHERE email = ?", (email.lower(),)).fetchone()
    if existing:
        conn.close()
        return {"ok": False, "error": "Email already registered. Please log in."}

    conn.execute(
        "INSERT INTO users (full_name, email, password_hash, role, created_at) VALUES (?,?,?,?,?)",
        (full_name.strip(), email.lower().strip(), _hash_password(password), role, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()
    return {"ok": True}


def authenticate_user(email: str, password: str) -> dict:
    """Verify credentials. Returns user dict or {'ok': False, 'error': '...'}."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT id, full_name, email, password_hash, role, created_at FROM users WHERE email = ?",
        (email.lower().strip(),)
    ).fetchone()
    conn.close()

    if not row:
        return {"ok": False, "error": "No account found with that email."}
    if not _verify_password(password, row["password_hash"]):
        return {"ok": False, "error": "Incorrect password."}

    return {
        "ok": True,
        "id": row["id"],
        "full_name": row["full_name"],
        "email": row["email"],
        "role": row["role"],
        "created_at": row["created_at"],
    }


# ── Analyst Queue API ───────────────────────────────────────────────

def get_all_analysts() -> list:
    """Return ALL registered users so every newly created account appears in the assignment dropdown."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, full_name, email, role, created_at FROM users ORDER BY full_name"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_assignments() -> list:
    """Return all analyst assignments (admin view)."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM analyst_assignments ORDER BY risk_score DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_my_assignments(analyst_email: str) -> list:
    """Return assignments for a specific analyst."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM analyst_assignments WHERE analyst_email = ? ORDER BY risk_score DESC",
        (analyst_email.lower(),)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def assign_customer(customer_id: str, customer_name: str, risk_score: float,
                    risk_category: str, analyst_email: str, analyst_name: str) -> dict:
    """Assign or reassign a customer to an analyst."""
    conn = _get_conn()
    try:
        existing = conn.execute(
            "SELECT id FROM analyst_assignments WHERE customer_id = ?", (customer_id.strip(),)
        ).fetchone()
        if existing:
            conn.execute(
                "UPDATE analyst_assignments "
                "SET analyst_email=?, analyst_name=?, assigned_at=?, status='Active' "
                "WHERE customer_id=?",
                (analyst_email.lower().strip(), analyst_name.strip(),
                 datetime.now().isoformat(), customer_id.strip())
            )
        else:
            conn.execute(
                "INSERT INTO analyst_assignments "
                "(customer_id, customer_name, risk_score, risk_category, "
                " analyst_email, analyst_name, assigned_at, status) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (customer_id.strip(), customer_name.strip() or customer_id.strip(),
                 float(risk_score), risk_category,
                 analyst_email.lower().strip(), analyst_name.strip(),
                 datetime.now().isoformat(), "Active")
            )
        conn.commit()          # explicit commit — visible to all other connections
        return {"ok": True, "rows": conn.execute("SELECT changes()").fetchone()[0]}
    except Exception as e:
        conn.rollback()
        return {"ok": False, "error": str(e)}
    finally:
        conn.close()


def update_assignment_status(assignment_id: int, status: str, notes: str = "") -> dict:
    """Update status/notes of an assignment."""
    conn = _get_conn()
    try:
        conn.execute(
            "UPDATE analyst_assignments SET status=?, notes=? WHERE id=?",
            (status, notes, assignment_id)
        )
        conn.commit()
        changed = conn.execute("SELECT changes()").fetchone()[0]
        if changed == 0:
            return {"ok": False, "error": f"Assignment ID {assignment_id} not found."}
        return {"ok": True}
    except Exception as e:
        conn.rollback()
        return {"ok": False, "error": str(e)}
    finally:
        conn.close()


def get_queue_summary() -> dict:
    """Return queue summary statistics."""
    conn = _get_conn()
    total = conn.execute("SELECT COUNT(*) as c FROM analyst_assignments WHERE status='Active'").fetchone()["c"]
    critical = conn.execute(
        "SELECT COUNT(*) as c FROM analyst_assignments WHERE risk_category='CRITICAL' AND status='Active'"
    ).fetchone()["c"]
    high = conn.execute(
        "SELECT COUNT(*) as c FROM analyst_assignments WHERE risk_category='HIGH' AND status='Active'"
    ).fetchone()["c"]
    analysts = conn.execute(
        "SELECT COUNT(DISTINCT analyst_email) as c FROM analyst_assignments WHERE status='Active'"
    ).fetchone()["c"]
    conn.close()
    return {"total": total, "critical": critical, "high": high, "active_analysts": analysts}
