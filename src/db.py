"""
PostgreSQL Database Integration Layer
Connects to the local Docker container 'kxcvbnm/expense-reimbursement-postgres:latest'
Provides data persistence for extracted financial documents, metadata, and audit logs.
"""

import os
import json
import uuid
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List
import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger("expense_db")

DB_HOST = os.getenv("POSTGRES_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "112233")
DB_NAME = os.getenv("POSTGRES_DB", "expense_reimbursement_db")


def get_db_connection():
    """Create and return a new connection to the PostgreSQL database."""
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        dbname=DB_NAME,
        connect_timeout=3
    )


def check_db_status() -> Dict[str, Any]:
    """Check connectivity to PostgreSQL container and list existing tables."""
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public';")
        tables = [row["table_name"] for row in cur.fetchall()]
        cur.execute("SELECT count(*) as doc_count FROM documents;")
        doc_count = cur.fetchone()["doc_count"]
        cur.close()
        conn.close()
        return {
            "connected": True,
            "online": True,
            "host": DB_HOST,
            "port": DB_PORT,
            "database": DB_NAME,
            "tables": tables,
            "documents_count": doc_count,
            "saved_documents_count": doc_count,
            "message": "Connected to PostgreSQL container successfully"
        }
    except Exception as e:
        logger.warning(f"Database connection check failed: {e}")
        return {
            "connected": False,
            "online": False,
            "host": DB_HOST,
            "port": DB_PORT,
            "database": DB_NAME,
            "error": str(e),
            "documents_count": 0,
            "saved_documents_count": 0,
            "message": "Cannot connect to PostgreSQL container. Ensure docker container is running."
        }


def ensure_default_user(conn) -> str:
    """
    Ensure at least one user exists in the 'users' table to satisfy
    the foreign key constraint fk_disbursement_records_user.
    Returns the UUID string of the default user.
    """
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT id FROM users LIMIT 1;")
    user = cur.fetchone()
    if user:
        cur.close()
        return str(user["id"])

    # Seed default officer user
    user_id = str(uuid.uuid4())
    cur.execute(
        """
        INSERT INTO users (id, username, email, role, created_at)
        VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
        ON CONFLICT (username) DO NOTHING
        RETURNING id;
        """,
        (user_id, "system_officer", "officer@seminar.ac.th", "officer")
    )
    res = cur.fetchone()
    cur.close()
    if res:
        return str(res["id"])

    # Fallback retrieve if conflict occurred
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT id FROM users WHERE username = 'system_officer';")
    res = cur.fetchone()
    cur.close()
    return str(res["id"])


def save_disbursement_document(data: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """
    Persist an extracted and verified document into PostgreSQL:
    1. disbursement_records
    2. documents (with JSONB extracted_data)
    3. expense_items (if any)
    4. work_logs
    """
    payload = dict(data or {})
    payload.update(kwargs)

    metadata = payload.get("metadata") or {}
    embed_text = payload.get("embed_text") or ""
    chat_answers = payload.get("chat_answers") or {}
    source_filename = payload.get("filename") or metadata.get("source_filename") or "unnamed_document.png"
    doc_type = payload.get("document_type") or payload.get("doc_type") or metadata.get("document_type") or "principle_approval_request"

    # Extract monetary amounts
    try:
        total_amount = float(metadata.get("total_amount") or 0.0)
    except (ValueError, TypeError):
        total_amount = 0.0

    receiver = (metadata.get("vendor_or_requester") or metadata.get("customer_name") or "หน่วยงานผู้เบิก/ผู้รับเงิน")[:50]
    rec_type = doc_type[:50]

    reconcile = metadata.get("math_reconciliation") or {}
    is_balanced = reconcile.get("is_balanced", False)
    status = "RECONCILED" if is_balanced else "PENDING_APPROVAL"

    # Date parsing
    doc_date_str = metadata.get("doc_date_iso")
    req_date = datetime.now(timezone.utc)
    if doc_date_str:
        try:
            req_date = datetime.fromisoformat(doc_date_str).replace(tzinfo=timezone.utc)
        except Exception:
            pass
    due_date = req_date + timedelta(days=30)

    conn = get_db_connection()
    try:
        # 1. Ensure user
        user_id = ensure_default_user(conn)

        cur = conn.cursor(cursor_factory=RealDictCursor)

        # 2. Insert disbursement_records
        record_id = str(uuid.uuid4())
        cur.execute(
            """
            INSERT INTO disbursement_records (
                id, user_id, record_type, receiver_name, approved_amount,
                actual_expense, status, request_date, due_date, created_at, updated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            RETURNING id;
            """,
            (
                record_id, user_id, rec_type, receiver, total_amount,
                total_amount, status, req_date, due_date
            )
        )

        # 3. Insert documents
        document_id = str(uuid.uuid4())
        s3_path = f"local://documents/{record_id}/{source_filename}_{document_id[:8]}"
        extracted_data_payload = {
            "filter_metadata": metadata,
            "embed_text": embed_text,
            "chat_answers": chat_answers,
            "math_reconciliation": reconcile,
            "saved_at": datetime.now(timezone.utc).isoformat()
        }

        cur.execute(
            """
            INSERT INTO documents (
                id, record_id, filename, s3_path, document_type,
                classified_step, extracted_data, uploaded_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            RETURNING id;
            """,
            (
                document_id, record_id, source_filename[:150], s3_path[:300],
                doc_type[:50], "EXTRACTED_VERIFIED", json.dumps(extracted_data_payload, ensure_ascii=False)
            )
        )

        # 4. Insert expense_items if line item information is available
        expense_items_list = payload.get("expense_items") or []
        if not expense_items_list and "รายละเอียดค่าใช้จ่าย" in chat_answers:
            # Create a summary expense item if not broken down into multiple lines
            item_desc = str(chat_answers["รายละเอียดค่าใช้จ่าย"])[:50]
            expense_items_list = [{
                "item_name": item_desc,
                "quantity": 1,
                "unit_price": total_amount,
                "total_amount": total_amount,
                "expense_type": doc_type[:50]
            }]

        for item in expense_items_list:
            item_id = str(uuid.uuid4())
            item_name = str(item.get("item_name") or item.get("description") or "รายการค่าใช้จ่าย")[:50]
            try:
                qty = int(item.get("quantity") or 1)
            except (ValueError, TypeError):
                qty = 1
            try:
                u_price = float(item.get("unit_price") or total_amount)
            except (ValueError, TypeError):
                u_price = total_amount
            try:
                t_amt = float(item.get("total_amount") or (qty * u_price))
            except (ValueError, TypeError):
                t_amt = total_amount

            cur.execute(
                """
                INSERT INTO expense_items (
                    id, record_id, document_id, item_name, quantity,
                    unit_price, total_amount, expense_type
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
                """,
                (
                    item_id, record_id, document_id, item_name, qty,
                    u_price, t_amt, doc_type[:50]
                )
            )

        # 5. Insert work_logs
        log_id = str(uuid.uuid4())
        notes = f"OCR Extracted & Saved. Formula/Math Status: {reconcile.get('status', 'unverified')}. {reconcile.get('details', '')}"
        cur.execute(
            """
            INSERT INTO work_logs (
                id, record_id, changed_by, previous_status, new_status, notes, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP);
            """,
            (
                log_id, record_id, user_id, "NEW", status, notes[:500]
            )
        )

        conn.commit()
        cur.close()
        conn.close()

        return {
            "success": True,
            "message": "บันทึกข้อมูลเอกสารลงฐานข้อมูล PostgreSQL สำเร็จ",
            "record_id": record_id,
            "document_id": document_id,
            "status": status,
            "receiver_name": receiver,
            "total_amount": total_amount,
            "filename": source_filename,
            "database": DB_NAME
        }
    except Exception as e:
        conn.rollback()
        conn.close()
        logger.error(f"Error saving disbursement document to database: {e}", exc_info=True)
        raise e


def get_recent_saved_documents(limit: int = 10) -> List[Dict[str, Any]]:
    """Retrieve recent saved records with document and status info."""
    conn = get_db_connection()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            """
            SELECT 
                d.id AS document_id,
                d.filename,
                d.document_type,
                d.uploaded_at,
                r.id AS record_id,
                r.receiver_name,
                r.approved_amount,
                r.status
            FROM documents d
            JOIN disbursement_records r ON d.record_id = r.id
            ORDER BY d.uploaded_at DESC
            LIMIT %s;
            """,
            (limit,)
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
        result = []
        for r in rows:
            result.append({
                "document_id": str(r["document_id"]),
                "record_id": str(r["record_id"]),
                "filename": r["filename"],
                "document_type": r["document_type"],
                "receiver_name": r["receiver_name"],
                "approved_amount": float(r["approved_amount"]),
                "status": r["status"],
                "uploaded_at": r["uploaded_at"].isoformat() if r["uploaded_at"] else None
            })
        return result
    except Exception as e:
        conn.close()
        logger.warning(f"Error fetching recent documents: {e}")
        return []
