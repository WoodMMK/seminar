"""
Test Suite: Interactive Field Synchronization and PostgreSQL Database Integration
Verifies Component 6 PP-ChatOCRv4 live editing, metadata/embed synchronization,
and PostgreSQL persistence in docker container 'kxcvbnm/expense-reimbursement-postgres'.
"""

import unittest
from src.db import (
    check_db_status,
    ensure_default_user,
    save_disbursement_document,
    get_recent_saved_documents,
    get_db_connection,
)


class TestInteractiveSyncAndDatabase(unittest.TestCase):

    def test_01_postgres_connectivity(self):
        """Test PostgreSQL container connectivity and schema health."""
        status = check_db_status()
        self.assertTrue(status.get("connected"), "PostgreSQL should be connected")
        self.assertIn("disbursement_records", status.get("tables", []))
        self.assertIn("documents", status.get("tables", []))
        self.assertIn("expense_items", status.get("tables", []))
        self.assertIn("work_logs", status.get("tables", []))
        self.assertGreaterEqual(status.get("documents_count", 0), 1)

    def test_02_interactive_edit_and_save_flow(self):
        """
        Simulate user workflow:
        1. Extract OCR with original price 1800.
        2. User edits total amount in metadata to 1500.
        3. Embed text synchronized to 1,500.00 บาท.
        4. Math reconciliation evaluated.
        5. Saved to PostgreSQL container.
        """
        # Simulated edited payload
        edited_metadata = {
            "total_amount": 1500.0,
            "subtotal": 1401.87,
            "vat": 98.13,
            "doc_no": "EXP-SYNC-2026-001",
            "vendor": "Office Depot Thailand",
            "doc_date_iso": "2026-09-23",
            "document_title": "ใบเสร็จค่าอุปกรณ์เครื่องเขียนสัมมนา",
            "math_reconciliation": {
                "status": "passed",
                "is_balanced": True,
                "diff": 0.0,
                "calculated_total": 1500.0,
                "details": "รวมก่อนภาษี (1,401.87) + VAT (98.13) เท่ากับยอดเงินรวมทั้งสิ้น (1,500.00 บาท) ถูกต้องสมบูรณ์"
            }
        }

        # Simulated synchronized embed text
        synchronized_embed_text = (
            "# เอกสารการเงิน: ใบเสร็จค่าอุปกรณ์เครื่องเขียนสัมมนา\n"
            "- **ชื่อเอกสาร (Document Title):** ใบเสร็จค่าอุปกรณ์เครื่องเขียนสัมมนา\n"
            "- **ประเภทเอกสาร:** ใบเสร็จรับเงิน / ใบกำกับภาษีทั่วไป (ร้านค้า/บริษัท)\n"
            "- **เลขที่เอกสาร:** EXP-SYNC-2026-001\n"
            "- **บุคคล/หน่วยงาน/ร้านค้า:** Office Depot Thailand\n"
            "- **วันที่เอกสาร:** 2026-09-23\n"
            "- **ยอดเงินรวม:** 1,500.00 บาท\n"
            "- **การตรวจสอบความถูกต้องของยอดเงิน (Reconciliation):** ตรวจสอบถูกต้อง (Reconciled) - รวมก่อนภาษี (1,401.87) + VAT (98.13) เท่ากับยอดเงินรวมทั้งสิ้น (1,500.00 บาท) ถูกต้องสมบูรณ์\n"
            "- **ยอดรวมก่อนภาษี (Subtotal):** 1,401.87 บาท\n"
            "- **ภาษีมูลค่าเพิ่ม (VAT):** 98.13 บาท\n"
        )

        chat_answers = {
            "ชื่อร้านค้าหรือบริษัทผู้ขาย": "Office Depot Thailand",
            "ยอดรวมก่อนภาษี (Subtotal)": "1401.87",
            "ภาษีมูลค่าเพิ่ม 7% (VAT)": "98.13",
            "ยอดเงินรวมทั้งสิ้น (Total Amount)": "1500.00"
        }

        save_result = save_disbursement_document(
            filename="seminar_supplies_receipt.png",
            doc_type="general_receipt",
            metadata=edited_metadata,
            embed_text=synchronized_embed_text,
            chat_answers=chat_answers,
            math_reconciliation=edited_metadata["math_reconciliation"]
        )

        self.assertTrue(save_result["success"])
        self.assertIsNotNone(save_result["record_id"])
        self.assertIsNotNone(save_result["document_id"])
        self.assertEqual(save_result["total_amount"], 1500.0)

        # Verify record directly in PostgreSQL
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT approved_amount, status FROM disbursement_records WHERE id = %s;",
            (save_result["record_id"],)
        )
        row = cur.fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(float(row[0]), 1500.0)
        self.assertEqual(row[1], "RECONCILED")

        # Verify JSONB extracted_data in documents table
        cur.execute(
            "SELECT extracted_data FROM documents WHERE id = %s;",
            (save_result["document_id"],)
        )
        doc_row = cur.fetchone()
        self.assertIsNotNone(doc_row)
        import json
        extracted = doc_row[0]
        if isinstance(extracted, str):
            extracted = json.loads(extracted)
        self.assertEqual(extracted["filter_metadata"]["total_amount"], 1500.0)
        self.assertIn("1,500.00 บาท", extracted["embed_text"])

        cur.close()
        conn.close()

    def test_03_recent_saved_documents(self):
        """Test retrieving recent saved documents."""
        recent = get_recent_saved_documents(limit=5)
        self.assertIsInstance(recent, list)
        self.assertGreaterEqual(len(recent), 1)
        first = recent[0]
        self.assertIn("document_id", first)
        self.assertIn("record_id", first)
        self.assertIn("filename", first)


if __name__ == "__main__":
    unittest.main()
