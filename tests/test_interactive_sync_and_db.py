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
    get_available_principle_cases,
    find_smart_case_match,
    get_dossier_dashboard_data,
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

    def test_04_principle_case_linking_and_dashboard_traceability(self):
        """
        Verify end-to-end Principle-Centric Case Linking and Dossier Dashboard:
        1. Save Principle Approval document (Root Anchor) -> creates case dossier.
        2. Save Procurement Memo referencing the principle doc -> links to same case.
        3. Save General Store Receipt without explicit principle ref -> links via Smart Matching.
        4. Query Dashboard API -> confirms case has all 3 documents, aggregated actual expense, and traceability checklist.
        """
        import uuid
        unique_principle_no = f"อว 0602/{uuid.uuid4().hex[:6]}"

        # Step 1: Save Root Principle Approval Document
        p_meta = {
            "doc_no": unique_principle_no,
            "principle_doc_no": unique_principle_no,
            "document_title": f"โครงการสัมมนาเชิงปฏิบัติการ AI ({unique_principle_no})",
            "document_type": "principle_approval_request",
            "document_type_name": "เอกสารขออนุมัติหลักการ",
            "total_amount": 50000.0,
            "vendor_or_requester": "ภาควิชาวิศวกรรมคอมพิวเตอร์",
            "doc_date_iso": "2026-09-01",
            "math_reconciliation": {"status": "passed", "is_balanced": True, "details": "วงเงิน 50,000.00 บาท"}
        }
        res_principle = save_disbursement_document(
            filename="01_principle_approval.png",
            doc_type="principle_approval_request",
            metadata=p_meta,
            embed_text=f"# เอกสารการเงิน: ขออนุมัติหลักการ {unique_principle_no}\n- วงเงิน: 50,000.00 บาท",
            chat_answers={"เรื่อง": p_meta["document_title"], "วงเงินงบประมาณ": "50000"}
        )
        self.assertTrue(res_principle["success"])
        case_id = res_principle["record_id"]
        self.assertIsNotNone(case_id)

        # Verify case was created
        cases = get_available_principle_cases()
        found_case = next((c for c in cases if c["id"] == case_id), None)
        self.assertIsNotNone(found_case)
        self.assertEqual(found_case["principle_doc_no"], unique_principle_no)
        self.assertEqual(found_case["approved_amount"], 50000.0)
        self.assertEqual(found_case["actual_expense"], 0.0)
        self.assertEqual(found_case["status"], "APPROVED_PRINCIPLE")

        # Step 2: Save Procurement Memo linking to the principle document
        proc_meta = {
            "doc_no": f"จัดหา-{uuid.uuid4().hex[:4]}",
            "principle_doc_no": unique_principle_no,
            "document_title": "ขออนุมัติจัดหาวัสดุอุปกรณ์สัมมนา",
            "document_type": "procurement_approval_request",
            "document_type_name": "ขออนุมัติจัดหาพัสดุ",
            "total_amount": 12000.0,
            "vendor_or_requester": "คณะกรรมการจัดหาพัสดุ",
            "doc_date_iso": "2026-09-05",
            "math_reconciliation": {"status": "passed", "is_balanced": True, "details": "ยอดจัดหา 12,000.00 บาท"}
        }
        res_proc = save_disbursement_document(
            filename="02_procurement_memo.png",
            doc_type="procurement_approval_request",
            metadata=proc_meta,
            embed_text=f"# ขออนุมัติจัดหาพัสดุ\n- อ้างอิงหลักการ: {unique_principle_no}\n- ยอดเงิน: 12,000.00 บาท",
            chat_answers={"อ้างอิงหลักการ": unique_principle_no, "ยอดรวม": "12000"},
            target_record_id=case_id
        )
        self.assertTrue(res_proc["success"])
        self.assertEqual(res_proc["record_id"], case_id)
        self.assertTrue(res_proc["attached_to_existing"])

        # Step 3: Test Smart Candidate Matching for unreferenced store receipt
        match = find_smart_case_match(
            amount=12000.0,
            vendor="ภาควิชาวิศวกรรมคอมพิวเตอร์",
            principle_doc_no=None
        )
        self.assertIsNotNone(match.get("matched_case_id"))
        self.assertGreater(match.get("confidence", 0), 0.5)

        # Save store receipt linking to the case
        rec_meta = {
            "doc_no": f"REC-SHOP-{uuid.uuid4().hex[:4]}",
            "principle_doc_no": unique_principle_no,
            "document_title": "ใบเสร็จรับเงินค่าวัสดุอุปกรณ์",
            "document_type": "general_receipt",
            "document_type_name": "ใบเสร็จรับเงินทั่วไป",
            "total_amount": 12000.0,
            "vendor_or_requester": "บริษัท อุปกรณ์ไอที จำกัด",
            "doc_date_iso": "2026-09-10",
            "math_reconciliation": {"status": "passed", "is_balanced": True, "details": "12,000.00 บาท"}
        }
        res_rec = save_disbursement_document(
            filename="03_it_shop_receipt.png",
            doc_type="general_receipt",
            metadata=rec_meta,
            embed_text=f"# ใบเสร็จรับเงินร้านค้า\n- อ้างอิงหลักการ: {unique_principle_no}\n- ยอดเงิน: 12,000.00 บาท",
            chat_answers={"ยอดเงินรวมทั้งสิ้น": "12000"},
            target_record_id=case_id
        )
        self.assertTrue(res_rec["success"])
        self.assertEqual(res_rec["record_id"], case_id)

        # Step 4: Verify Dashboard Grouping & Traceability
        dashboard = get_dossier_dashboard_data()
        self.assertIn("summary", dashboard)
        self.assertIn("cases", dashboard)
        self.assertGreaterEqual(dashboard["summary"]["total_cases"], 1)

        # Find our created case in dashboard
        dossier = next((c for c in dashboard["cases"] if c["record_id"] == case_id), None)
        self.assertIsNotNone(dossier, "Dossier case should be present in dashboard")
        self.assertEqual(dossier["principle_doc_no"], unique_principle_no)
        self.assertEqual(dossier["approved_amount"], 50000.0)
        # Non-principle docs sum = 12000 (procurement) + 12000 (receipt) = 24000
        self.assertEqual(dossier["actual_expense"], 24000.0)
        self.assertEqual(dossier["remaining_amount"], 26000.0)
        self.assertEqual(dossier["status"], "RECONCILED")  # 24000 <= 50000

        # Verify child documents attached to this case
        docs = dossier["documents"]
        self.assertEqual(len(docs), 3, "Case should have exactly 3 attached documents")

        # Verify Traceability checklist flags
        chk = dossier["checklist"]
        self.assertTrue(chk["has_principle"], "Checklist should have principle document")
        self.assertTrue(chk["has_procurement"], "Checklist should have procurement document")
        self.assertTrue(chk["has_receipt"], "Checklist should have receipt document")


if __name__ == "__main__":
    unittest.main()

