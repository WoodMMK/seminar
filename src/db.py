"""
PostgreSQL Database Integration Layer (Component 6)
Connects to the local Docker container 'kxcvbnm/expense-reimbursement-postgres:latest'
Provides data persistence for extracted financial documents, metadata, audit logs,
and Principle-Centric Case Linking & Traceability Dossiers.
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
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        dbname=DB_NAME,
        connect_timeout=3
    )
    return conn


def ensure_schema_columns(conn):
    """Ensure principle_doc_no and title columns exist on disbursement_records."""
    try:
        cur = conn.cursor()
        cur.execute(
            """
            ALTER TABLE disbursement_records ADD COLUMN IF NOT EXISTS principle_doc_no VARCHAR(100);
            ALTER TABLE disbursement_records ADD COLUMN IF NOT EXISTS title VARCHAR(255);
            CREATE INDEX IF NOT EXISTS idx_disbursement_records_principle ON disbursement_records (principle_doc_no);
            """
        )
        conn.commit()
        cur.close()
    except Exception as e:
        conn.rollback()
        logger.warning(f"Could not ensure schema columns: {e}")


def check_db_status() -> Dict[str, Any]:
    """Check connectivity to PostgreSQL container and list existing tables and counts."""
    try:
        conn = get_db_connection()
        ensure_schema_columns(conn)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public';")
        tables = [row["table_name"] for row in cur.fetchall()]
        cur.execute("SELECT count(*) as doc_count FROM documents;")
        doc_count = cur.fetchone()["doc_count"]
        cur.execute("SELECT count(*) as case_count FROM disbursement_records;")
        case_count = cur.fetchone()["case_count"]
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
            "cases_count": case_count,
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
            "cases_count": 0,
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


def get_available_principle_cases() -> List[Dict[str, Any]]:
    """
    Retrieve all disbursement cases for UI selection.
    Returns list of cases with principle document numbers, titles, and budget info.
    """
    conn = get_db_connection()
    ensure_schema_columns(conn)
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            """
            SELECT 
                r.id,
                COALESCE(r.principle_doc_no, '-') AS principle_doc_no,
                COALESCE(r.title, r.receiver_name, 'ชุดเรื่องเบิกจ่าย') AS title,
                r.receiver_name,
                r.approved_amount,
                r.actual_expense,
                r.status,
                r.created_at,
                COUNT(d.id) AS document_count
            FROM disbursement_records r
            LEFT JOIN documents d ON r.id = d.record_id
            GROUP BY r.id, r.principle_doc_no, r.title, r.receiver_name, r.approved_amount, r.actual_expense, r.status, r.created_at
            ORDER BY r.created_at DESC;
            """
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
        cases = []
        for r in rows:
            cases.append({
                "id": str(r["id"]),
                "principle_doc_no": r["principle_doc_no"],
                "title": r["title"],
                "receiver_name": r["receiver_name"],
                "approved_amount": float(r["approved_amount"]),
                "actual_expense": float(r["actual_expense"]),
                "status": r["status"],
                "document_count": int(r["document_count"]),
                "created_at": r["created_at"].isoformat() if r["created_at"] else None
            })
        return cases
    except Exception as e:
        conn.close()
        logger.warning(f"Error fetching principle cases: {e}")
        return []


def find_smart_case_match(
    amount: Optional[float] = None,
    vendor: Optional[str] = None,
    principle_doc_no: Optional[str] = None
) -> Dict[str, Any]:
    """
    Smart Candidate Matching Algorithm for linking documents to cases:
    1. Exact Match: Matches principle_doc_no with existing case (Confidence: 100%)
    2. Amount Match: Matches amount with child expense item / line item (Confidence: 85-90%)
    3. Vendor / Keyword Match: Matches vendor name with case title/items (Confidence: 75%)
    4. Fallback: Most recent active case (Confidence: 50%)
    """
    cases = get_available_principle_cases()
    if not cases:
        return {"matched_case_id": None, "confidence": 0, "reason": "ยังไม่มีชุดเรื่องในฐานข้อมูล"}

    # 1. Exact principle_doc_no match
    if principle_doc_no and principle_doc_no.strip() and principle_doc_no.strip() not in ("-", "ไม่มีระบุ"):
        clean_pno = principle_doc_no.strip()
        for c in cases:
            if c["principle_doc_no"] and clean_pno.lower() in c["principle_doc_no"].lower():
                return {
                    "matched_case_id": c["id"],
                    "case_title": c["title"],
                    "principle_doc_no": c["principle_doc_no"],
                    "confidence": 1.0,
                    "reason": f"ตรวจพบเลขอ้างอิงหลักการตรงกัน ({c['principle_doc_no']})"
                }

    # 2. Amount match in line items or difference
    if amount and amount > 0:
        for c in cases:
            # Check if amount matches approved or remaining
            if abs(c["approved_amount"] - amount) <= 1.0:
                return {
                    "matched_case_id": c["id"],
                    "case_title": c["title"],
                    "principle_doc_no": c["principle_doc_no"],
                    "confidence": 0.88,
                    "reason": f"ยอดเงิน {amount:,.2f} บาท ตรงกับวงเงินในชุดเรื่อง '{c['title']}'"
                }

    # 3. Vendor match
    if vendor and len(vendor.strip()) > 3:
        clean_v = vendor.strip().lower()
        for c in cases:
            if clean_v in c["receiver_name"].lower() or clean_v in c["title"].lower():
                return {
                    "matched_case_id": c["id"],
                    "case_title": c["title"],
                    "principle_doc_no": c["principle_doc_no"],
                    "confidence": 0.75,
                    "reason": f"ชื่อบุคคล/ร้านค้าสอดคล้องกับชุดเรื่อง '{c['title']}'"
                }

    # 4. Fallback: Most recent active case
    latest = cases[0]
    return {
        "matched_case_id": latest["id"],
        "case_title": latest["title"],
        "principle_doc_no": latest["principle_doc_no"],
        "confidence": 0.50,
        "reason": f"แนะนำตามชุดเรื่องล่าสุดที่กำลังดำเนินการ ('{latest['title']}')"
    }


def save_disbursement_document(data: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """
    Persist an extracted and verified document into PostgreSQL:
    - If target_record_id or matching principle_doc_no provided: links into existing case dossier.
    - If new case or principle approval request: creates root case in disbursement_records.
    - Inserts into documents with JSONB extracted_data.
    - Inserts into expense_items.
    - Updates actual_expense and status in disbursement_records.
    - Logs into work_logs.
    """
    payload = dict(data or {})
    payload.update(kwargs)

    metadata = payload.get("metadata") or {}
    embed_text = payload.get("embed_text") or ""
    chat_answers = payload.get("chat_answers") or {}
    source_filename = payload.get("filename") or metadata.get("source_filename") or "unnamed_document.png"
    doc_type = payload.get("document_type") or payload.get("doc_type") or metadata.get("document_type") or "principle_approval_request"

    # Principle Document Reference Linking
    target_record_id = payload.get("target_record_id")
    is_new_case = payload.get("is_new_case", False)
    case_title = payload.get("case_title")
    principle_doc_no = (
        payload.get("principle_doc_no")
        or metadata.get("principle_doc_no")
        or (metadata.get("doc_no") if doc_type == "principle_approval_request" else None)
    )

    # Extract monetary amounts
    try:
        total_amount = float(metadata.get("total_amount") or 0.0)
    except (ValueError, TypeError):
        total_amount = 0.0

    receiver = (metadata.get("vendor_or_requester") or metadata.get("customer_name") or "หน่วยงานผู้เบิก/ผู้รับเงิน")[:50]
    rec_type = doc_type[:50]

    reconcile = metadata.get("math_reconciliation") or {}
    is_balanced = reconcile.get("is_balanced", False)

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
    ensure_schema_columns(conn)

    try:
        user_id = ensure_default_user(conn)
        cur = conn.cursor(cursor_factory=RealDictCursor)

        record_id = None
        attached_to_existing = False

        # 1. Determine whether to attach to existing case or create a new case
        if not is_new_case and target_record_id and target_record_id != "__new__":
            cur.execute("SELECT id, approved_amount, actual_expense, principle_doc_no, title FROM disbursement_records WHERE id = %s;", (target_record_id,))
            existing = cur.fetchone()
            if existing:
                record_id = str(existing["id"])
                attached_to_existing = True

        # Auto-match by principle_doc_no if not explicit
        if not record_id and not is_new_case and principle_doc_no and doc_type != "principle_approval_request":
            cur.execute("SELECT id, approved_amount, actual_expense, principle_doc_no, title FROM disbursement_records WHERE principle_doc_no = %s LIMIT 1;", (principle_doc_no.strip(),))
            match_row = cur.fetchone()
            if match_row:
                record_id = str(match_row["id"])
                attached_to_existing = True

        # If still no record_id -> Create New Case Dossier in disbursement_records
        if not record_id:
            record_id = str(uuid.uuid4())
            if doc_type == "principle_approval_request":
                approved_ceiling = total_amount
                actual_spent = 0.0
                status = "APPROVED_PRINCIPLE"
                p_no = principle_doc_no or metadata.get("doc_no") or "อว-หลักการ"
                c_title = case_title or metadata.get("document_title") or f"ขออนุมัติหลักการ ({p_no})"
            else:
                approved_ceiling = total_amount
                actual_spent = total_amount
                status = "RECONCILED" if is_balanced else "PENDING_APPROVAL"
                p_no = principle_doc_no or "รอดำเนินการ"
                c_title = case_title or metadata.get("document_title") or f"ชุดเรื่องเบิกจ่าย ({p_no})"

            cur.execute(
                """
                INSERT INTO disbursement_records (
                    id, user_id, record_type, receiver_name, approved_amount,
                    actual_expense, status, request_date, due_date, principle_doc_no, title, created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);
                """,
                (
                    record_id, user_id, rec_type, receiver, approved_ceiling,
                    actual_spent, status, req_date, due_date, p_no, c_title
                )
            )
        else:
            # If attaching a principle approval document to existing case, update budget ceiling and principle_no
            if doc_type == "principle_approval_request":
                cur.execute(
                    """
                    UPDATE disbursement_records 
                    SET approved_amount = %s,
                        principle_doc_no = COALESCE(%s, principle_doc_no),
                        title = COALESCE(%s, title),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s;
                    """,
                    (total_amount, principle_doc_no, case_title, record_id)
                )

        # 2. Insert into documents table
        document_id = str(uuid.uuid4())
        s3_path = f"local://documents/{record_id}/{source_filename}_{document_id[:8]}"
        extracted_data_payload = {
            "filter_metadata": metadata,
            "embed_text": embed_text,
            "chat_answers": chat_answers,
            "math_reconciliation": reconcile,
            "principle_doc_no": principle_doc_no,
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

        # 3. Insert expense_items
        expense_items_list = payload.get("expense_items") or []
        if not expense_items_list and "รายละเอียดค่าใช้จ่าย" in chat_answers:
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

        # 4. Recalculate actual_expense and status for the parent case
        if attached_to_existing:
            # Sum amounts from all non-principle documents attached to this case
            cur.execute(
                """
                SELECT COALESCE(SUM(CAST(extracted_data->'filter_metadata'->>'total_amount' AS NUMERIC)), 0) as total_spent
                FROM documents
                WHERE record_id = %s AND document_type != 'principle_approval_request';
                """,
                (record_id,)
            )
            spent_row = cur.fetchone()
            total_spent = float(spent_row["total_spent"]) if spent_row else total_amount

            cur.execute("SELECT approved_amount FROM disbursement_records WHERE id = %s;", (record_id,))
            case_row = cur.fetchone()
            approved_amount = float(case_row["approved_amount"]) if case_row else 0.0

            if approved_amount > 0 and total_spent > approved_amount:
                case_status = "OVER_BUDGET"
            elif approved_amount > 0 and total_spent <= approved_amount:
                case_status = "RECONCILED"
            else:
                case_status = "PENDING_APPROVAL"

            cur.execute(
                """
                UPDATE disbursement_records
                SET actual_expense = %s,
                    status = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s;
                """,
                (total_spent, case_status, record_id)
            )

        # 5. Insert work_logs
        log_id = str(uuid.uuid4())
        notes = f"Doc: {source_filename} [{doc_type}]. Principle Ref: {principle_doc_no or '-'}. Linked to case: {record_id}."
        cur.execute(
            """
            INSERT INTO work_logs (
                id, record_id, changed_by, previous_status, new_status, notes, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP);
            """,
            (
                log_id, record_id, user_id, "PREVIOUS", "EXTRACTED_VERIFIED", notes[:500]
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
            "principle_doc_no": principle_doc_no,
            "attached_to_existing": attached_to_existing,
            "total_amount": total_amount,
            "filename": source_filename,
            "database": DB_NAME
        }
    except Exception as e:
        conn.rollback()
        conn.close()
        logger.error(f"Error saving disbursement document to database: {e}", exc_info=True)
        raise e


def get_dossier_dashboard_data() -> Dict[str, Any]:
    """
    Retrieve all disbursement case dossiers grouped with their attached documents,
    budget comparison (Approved vs Actual), and Traceability checklist.
    """
    conn = get_db_connection()
    ensure_schema_columns(conn)
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        # Fetch cases
        cur.execute(
            """
            SELECT 
                r.id,
                COALESCE(r.principle_doc_no, '-') AS principle_doc_no,
                COALESCE(r.title, r.receiver_name, 'ชุดเรื่องเบิกจ่าย') AS title,
                r.receiver_name,
                r.approved_amount,
                r.actual_expense,
                r.status,
                r.created_at,
                r.updated_at
            FROM disbursement_records r
            ORDER BY r.created_at DESC;
            """
        )
        case_rows = cur.fetchall()

        total_cases = len(case_rows)
        total_docs = 0
        total_approved = 0.0
        total_actual = 0.0

        cases_data = []

        for c in case_rows:
            case_id = str(c["id"])
            approved = float(c["approved_amount"])
            actual = float(c["actual_expense"])
            remaining = approved - actual
            total_approved += approved
            total_actual += actual

            # Fetch child documents
            cur.execute(
                """
                SELECT 
                    id, filename, document_type, uploaded_at, extracted_data
                FROM documents
                WHERE record_id = %s
                ORDER BY uploaded_at ASC;
                """,
                (case_id,)
            )
            doc_rows = cur.fetchall()
            total_docs += len(doc_rows)

            docs_list = []
            has_principle = False
            has_procurement = False
            has_receipt = False
            has_disbursement = False

            for d in doc_rows:
                raw_payload = d["extracted_data"] or {}
                if isinstance(raw_payload, str):
                    try:
                        raw_payload = json.loads(raw_payload)
                    except Exception:
                        raw_payload = {}

                meta = raw_payload.get("filter_metadata") or {}
                dtype = d["document_type"]

                if "principle" in dtype:
                    has_principle = True
                if "procurement" in dtype:
                    has_procurement = True
                if "receipt" in dtype or "voucher" in dtype:
                    has_receipt = True
                if "disbursement" in dtype or "advance" in dtype:
                    has_disbursement = True

                docs_list.append({
                    "document_id": str(d["id"]),
                    "filename": d["filename"],
                    "document_type": dtype,
                    "document_type_name": meta.get("document_type_name") or dtype,
                    "doc_no": meta.get("doc_no") or meta.get("document_no") or "-",
                    "doc_date_iso": meta.get("doc_date_iso") or meta.get("date") or "-",
                    "party_name": meta.get("vendor_or_requester") or meta.get("vendor") or "-",
                    "total_amount": float(meta.get("total_amount") or 0.0),
                    "subtotal": float(meta.get("subtotal") or 0.0) if meta.get("subtotal") is not None else None,
                    "vat": float(meta.get("vat") or 0.0) if meta.get("vat") is not None else None,
                    "reconciliation_status": meta.get("math_reconciliation", {}).get("status") or "unverified",
                    "reconciliation_details": meta.get("math_reconciliation", {}).get("details") or "",
                    "uploaded_at": d["uploaded_at"].isoformat() if d["uploaded_at"] else None,
                    "extracted_data": raw_payload
                })

            cases_data.append({
                "record_id": case_id,
                "principle_doc_no": c["principle_doc_no"],
                "title": c["title"],
                "receiver_name": c["receiver_name"],
                "approved_amount": approved,
                "actual_expense": actual,
                "remaining_amount": remaining,
                "status": c["status"],
                "created_at": c["created_at"].isoformat() if c["created_at"] else None,
                "updated_at": c["updated_at"].isoformat() if c["updated_at"] else None,
                "checklist": {
                    "has_principle": has_principle,
                    "has_procurement": has_procurement,
                    "has_receipt": has_receipt,
                    "has_disbursement": has_disbursement
                },
                "documents": docs_list
            })

        cur.close()
        conn.close()

        return {
            "summary": {
                "total_cases": total_cases,
                "total_documents": total_docs,
                "total_approved_budget": round(total_approved, 2),
                "total_actual_expense": round(total_actual, 2),
                "total_remaining_budget": round(total_approved - total_actual, 2)
            },
            "cases": cases_data
        }
    except Exception as e:
        conn.close()
        logger.error(f"Error fetching dashboard data: {e}", exc_info=True)
        return {
            "summary": {
                "total_cases": 0,
                "total_documents": 0,
                "total_approved_budget": 0.0,
                "total_actual_expense": 0.0,
                "total_remaining_budget": 0.0
            },
            "cases": []
        }


def get_recent_saved_documents(limit: int = 10) -> List[Dict[str, Any]]:
    """Retrieve recent saved records with document and status info."""
    conn = get_db_connection()
    ensure_schema_columns(conn)
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
                COALESCE(r.principle_doc_no, '-') AS principle_doc_no,
                COALESCE(r.title, r.receiver_name, 'ชุดเรื่องเบิกจ่าย') AS title,
                r.receiver_name,
                r.approved_amount,
                r.actual_expense,
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
                "principle_doc_no": r["principle_doc_no"],
                "title": r["title"],
                "filename": r["filename"],
                "document_type": r["document_type"],
                "receiver_name": r["receiver_name"],
                "approved_amount": float(r["approved_amount"]),
                "actual_expense": float(r["actual_expense"]),
                "status": r["status"],
                "uploaded_at": r["uploaded_at"].isoformat() if r["uploaded_at"] else None
            })
        return result
    except Exception as e:
        conn.close()
        logger.warning(f"Error fetching recent documents: {e}")
        return []
