"""
PDIE Real Scheduler Module
Schedules actual reminders using APScheduler with SQLite persistence.
Jobs fire at the scheduled time and send real messages via real_messaging.

Author: PDIE Team | Barclays Hack-O-Hire 2026
"""

import os
import json
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path

# Database path
DB_PATH = Path(__file__).parent / 'pdie_reminders.db'

# Lock for thread safety
_db_lock = threading.Lock()


def _init_db():
    """Initialize the SQLite database for reminders."""
    with _db_lock:
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute('''
            CREATE TABLE IF NOT EXISTS reminders (
                reminder_id TEXT PRIMARY KEY,
                customer_id TEXT NOT NULL,
                customer_phone TEXT NOT NULL,
                channel TEXT NOT NULL,
                message TEXT NOT NULL,
                reminder_type TEXT NOT NULL,
                scheduled_for TEXT NOT NULL,
                created_at TEXT NOT NULL,
                status TEXT DEFAULT 'SCHEDULED',
                executed_at TEXT,
                result TEXT
            )
        ''')
        conn.commit()
        conn.close()


def schedule_reminder(
    customer_id: str,
    phone: str,
    message: str,
    channel: str,
    delay_hours: int,
    reminder_type: str = 'follow_up'
) -> Dict[str, Any]:
    """
    Schedule a real reminder that will fire after delay_hours.
    The reminder is persisted in SQLite so it survives server restarts.
    
    Args:
        customer_id: Customer identifier
        phone: Phone number (E.164 format)
        message: Message to send when reminder fires
        channel: 'SMS' or 'WhatsApp'
        delay_hours: Hours from now to send the reminder
        reminder_type: 'follow_up', 'escalation', or 'final_notice'
        
    Returns:
        Dict with reminder_id, scheduled_for, and status
    """
    
    _init_db()
    
    reminder_id = f'REM-{uuid.uuid4().hex[:8].upper()}'
    scheduled_for = datetime.now() + timedelta(hours=delay_hours)
    created_at = datetime.now().isoformat()
    
    with _db_lock:
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute('''
            INSERT INTO reminders (reminder_id, customer_id, customer_phone, channel, 
                                   message, reminder_type, scheduled_for, created_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'SCHEDULED')
        ''', (
            reminder_id, customer_id, phone, channel,
            message, reminder_type, scheduled_for.isoformat(), created_at
        ))
        conn.commit()
        conn.close()
    
    # Try to set up APScheduler background job
    _schedule_apscheduler_job(reminder_id, phone, message, channel, scheduled_for)
    
    return {
        'success': True,
        'reminder_id': reminder_id,
        'customer_id': customer_id,
        'phone': phone,
        'channel': channel,
        'reminder_type': reminder_type,
        'scheduled_for': scheduled_for.strftime('%Y-%m-%d %H:%M'),
        'hours_from_now': delay_hours,
        'status': 'SCHEDULED',
        'persisted': True,
        'auto_cancel_if_responded': True,
    }


def _schedule_apscheduler_job(reminder_id: str, phone: str, message: str, channel: str, run_at: datetime):
    """Try to schedule an APScheduler job. Graceful if APScheduler not available."""
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
        
        scheduler = _get_scheduler()
        if scheduler and scheduler.running:
            scheduler.add_job(
                _execute_reminder,
                'date',
                run_date=run_at,
                args=[reminder_id, phone, message, channel],
                id=reminder_id,
                replace_existing=True,
                misfire_grace_time=3600,  # 1 hour grace
            )
    except ImportError:
        pass  # APScheduler not installed — reminders are still in DB
    except Exception:
        pass


# Singleton scheduler
_scheduler_instance = None
_scheduler_lock = threading.Lock()


def _get_scheduler():
    """Get or create the singleton scheduler."""
    global _scheduler_instance
    if _scheduler_instance is not None:
        return _scheduler_instance
    
    with _scheduler_lock:
        if _scheduler_instance is not None:
            return _scheduler_instance
        
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
            
            jobstores = {
                'default': SQLAlchemyJobStore(url=f'sqlite:///{DB_PATH}')
            }
            _scheduler_instance = BackgroundScheduler(jobstores=jobstores)
            _scheduler_instance.start()
            return _scheduler_instance
        except ImportError:
            return None
        except Exception:
            return None


def _execute_reminder(reminder_id: str, phone: str, message: str, channel: str):
    """Execute a scheduled reminder — sends the actual message."""
    import real_messaging
    
    result = real_messaging.send_message(phone, message, channel)
    
    # Update DB status
    with _db_lock:
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute('''
            UPDATE reminders SET status = ?, executed_at = ?, result = ?
            WHERE reminder_id = ?
        ''', (
            'EXECUTED' if result.get('success') else 'FAILED',
            datetime.now().isoformat(),
            json.dumps(result, default=str),
            reminder_id
        ))
        conn.commit()
        conn.close()
    
    return result


def get_pending_reminders(customer_id: str = None) -> List[Dict[str, Any]]:
    """Get all pending (not yet executed) reminders, optionally filtered by customer."""
    _init_db()
    
    with _db_lock:
        conn = sqlite3.connect(str(DB_PATH))
        if customer_id:
            cursor = conn.execute(
                'SELECT * FROM reminders WHERE status = ? AND customer_id = ? ORDER BY scheduled_for',
                ('SCHEDULED', customer_id)
            )
        else:
            cursor = conn.execute(
                'SELECT * FROM reminders WHERE status = ? ORDER BY scheduled_for',
                ('SCHEDULED',)
            )
        
        columns = [d[0] for d in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        conn.close()
    
    return rows


def get_all_reminders(customer_id: str = None) -> List[Dict[str, Any]]:
    """Get all reminders (all statuses), optionally filtered by customer."""
    _init_db()
    
    with _db_lock:
        conn = sqlite3.connect(str(DB_PATH))
        if customer_id:
            cursor = conn.execute(
                'SELECT * FROM reminders WHERE customer_id = ? ORDER BY scheduled_for DESC',
                (customer_id,)
            )
        else:
            cursor = conn.execute(
                'SELECT * FROM reminders ORDER BY scheduled_for DESC'
            )
        
        columns = [d[0] for d in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        conn.close()
    
    return rows


def cancel_reminder(reminder_id: str) -> Dict[str, Any]:
    """Cancel a scheduled reminder."""
    _init_db()
    
    with _db_lock:
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute(
            'UPDATE reminders SET status = ? WHERE reminder_id = ? AND status = ?',
            ('CANCELLED', reminder_id, 'SCHEDULED')
        )
        conn.commit()
        conn.close()
    
    # Also remove from APScheduler
    try:
        scheduler = _get_scheduler()
        if scheduler and scheduler.running:
            scheduler.remove_job(reminder_id)
    except Exception:
        pass
    
    return {
        'success': True,
        'reminder_id': reminder_id,
        'status': 'CANCELLED',
    }


def cancel_all_for_customer(customer_id: str) -> Dict[str, Any]:
    """Cancel all pending reminders for a customer (e.g., when they respond)."""
    _init_db()
    
    with _db_lock:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.execute(
            'SELECT reminder_id FROM reminders WHERE customer_id = ? AND status = ?',
            (customer_id, 'SCHEDULED')
        )
        reminder_ids = [row[0] for row in cursor.fetchall()]
        
        conn.execute(
            'UPDATE reminders SET status = ? WHERE customer_id = ? AND status = ?',
            ('CANCELLED', customer_id, 'SCHEDULED')
        )
        conn.commit()
        conn.close()
    
    # Remove from APScheduler
    for rid in reminder_ids:
        try:
            scheduler = _get_scheduler()
            if scheduler and scheduler.running:
                scheduler.remove_job(rid)
        except Exception:
            pass
    
    return {
        'success': True,
        'customer_id': customer_id,
        'cancelled_count': len(reminder_ids),
        'cancelled_ids': reminder_ids,
    }
