import sqlite3
import json
import os
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "deepfake_audit.db")

def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    
    # Create scans table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id TEXT PRIMARY KEY,
            media_type TEXT NOT NULL,
            filename TEXT NOT NULL,
            is_fake INTEGER NOT NULL,
            confidence_score REAL NOT NULL,
            verdict TEXT NOT NULL,
            processing_time_ms REAL NOT NULL,
            details_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    
    # Create feedback table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id TEXT PRIMARY KEY,
            scan_id TEXT NOT NULL,
            user_verdict TEXT NOT NULL,
            notes TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (scan_id) REFERENCES scans (id)
        )
    """)
    
    conn.commit()
    conn.close()

def log_scan(
    media_type: str,
    filename: str,
    is_fake: bool,
    confidence_score: float,
    verdict: str,
    processing_time_ms: float,
    details: Dict[str, Any]
) -> str:
    scan_id = str(uuid.uuid4())[:8]
    created_at = datetime.utcnow().isoformat()
    
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO scans (id, media_type, filename, is_fake, confidence_score, verdict, processing_time_ms, details_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        scan_id,
        media_type,
        filename,
        1 if is_fake else 0,
        confidence_score,
        verdict,
        processing_time_ms,
        json.dumps(details),
        created_at
    ))
    conn.commit()
    conn.close()
    return scan_id

def log_user_feedback(scan_id: str, user_verdict: str, notes: str = "") -> str:
    feedback_id = str(uuid.uuid4())[:8]
    created_at = datetime.utcnow().isoformat()
    
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO feedback (id, scan_id, user_verdict, notes, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (
        feedback_id,
        scan_id,
        user_verdict,
        notes,
        created_at
    ))
    conn.commit()
    conn.close()
    return feedback_id

def get_recent_scans(limit: int = 15) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT s.*, f.user_verdict as feedback_verdict, f.notes as feedback_notes
        FROM scans s
        LEFT JOIN feedback f ON s.id = f.scan_id
        ORDER BY s.created_at DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for row in rows:
        item = dict(row)
        item["is_fake"] = bool(item["is_fake"])
        try:
            item["details"] = json.loads(item["details_json"])
        except Exception:
            item["details"] = {}
        del item["details_json"]
        results.append(item)
    return results

def get_scan_statistics() -> Dict[str, Any]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as total FROM scans")
    total = cursor.fetchone()["total"]
    
    cursor.execute("SELECT COUNT(*) as fake_count FROM scans WHERE is_fake = 1")
    fake_count = cursor.fetchone()["fake_count"]
    
    cursor.execute("SELECT COUNT(*) as feedback_count FROM feedback")
    feedback_count = cursor.fetchone()["feedback_count"]
    
    conn.close()
    return {
        "total_scans": total,
        "flagged_fakes": fake_count,
        "flagged_authentic": total - fake_count,
        "feedback_submissions": feedback_count
    }
