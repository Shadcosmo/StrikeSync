import hashlib
import sqlite3
from datetime import datetime

import pandas as pd

from strikesync.paths import DB_PATH


def _connect() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def init_db() -> None:
    conn = _connect()
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)")
    c.execute("CREATE TABLE IF NOT EXISTS history (username TEXT, match_percentage REAL, timestamp TEXT)")
    columns_to_add = [
        ("dob", "TEXT"),
        ("weight", "REAL"),
        ("height", "REAL"),
        ("email", "TEXT"),
        ("phone", "TEXT"),
        ("security_q", "TEXT"),
        ("security_a", "TEXT"),
    ]
    for col_name, col_type in columns_to_add:
        try:
            c.execute(f"ALTER TABLE users ADD COLUMN {col_name} {col_type}")
        except sqlite3.OperationalError:
            pass
    conn.commit()
    conn.close()


def hash_password(password: str) -> str:
    return hashlib.sha256(str.encode(password)).hexdigest()


def add_user(username: str, password: str, sec_q: str, sec_a: str) -> bool:
    conn = _connect()
    c = conn.cursor()
    try:
        sec_a_hashed = hash_password(sec_a.strip().lower())
        c.execute(
            """INSERT INTO users
                     (username, password, dob, weight, height, email, phone, security_q, security_a)
                     VALUES (?, ?, "", 0.0, 0.0, "", "", ?, ?)""",
            (username, hash_password(password), sec_q, sec_a_hashed),
        )
        conn.commit()
        success = True
    except sqlite3.IntegrityError:
        success = False
    conn.close()
    return success


def login_user(username: str, password: str):
    conn = _connect()
    c = conn.cursor()
    c.execute(
        "SELECT username FROM users WHERE username = ? AND password = ?",
        (username, hash_password(password)),
    )
    data = c.fetchone()
    conn.close()
    return data


def get_user_for_recovery(username: str):
    conn = _connect()
    c = conn.cursor()
    c.execute(
        "SELECT username, security_q, security_a FROM users WHERE username = ?",
        (username,),
    )
    data = c.fetchone()
    conn.close()
    return data


def update_password(username: str, new_password: str) -> None:
    conn = _connect()
    c = conn.cursor()
    c.execute(
        "UPDATE users SET password = ? WHERE username = ?",
        (hash_password(new_password), username),
    )
    conn.commit()
    conn.close()


def update_profile(username: str, dob_str: str, weight: float, height: float) -> None:
    conn = _connect()
    c = conn.cursor()
    c.execute(
        "UPDATE users SET dob=?, weight=?, height=? WHERE username=?",
        (dob_str, weight, height, username),
    )
    conn.commit()
    conn.close()


def get_profile(username: str):
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT dob, weight, height FROM users WHERE username=?", (username,))
    data = c.fetchone()
    conn.close()
    return data


def add_history(username: str, score: float) -> None:
    conn = _connect()
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute(
        "INSERT INTO history (username, match_percentage, timestamp) VALUES (?, ?, ?)",
        (username, score, timestamp),
    )
    conn.commit()
    conn.close()


def get_history(username: str) -> pd.DataFrame:
    conn = _connect()
    df = pd.read_sql_query(
        'SELECT timestamp as Date, match_percentage as "Match (%)" FROM history WHERE username=? ORDER BY timestamp DESC',
        conn,
        params=(username,),
    )
    conn.close()
    return df
