"""
Test Suite: Hybrid RAG Chatbot & pgvector Integration (Component 7)
Verifies:
1. EmbeddingService (1024-dim dense vectors, L2 normalization)
2. Hybrid Query Intent Router:
   - EXHAUSTIVE_AGGREGATION (Solves the Top-K truncation problem)
   - CASE_DOSSIER_LOOKUP
   - RECONCILIATION_AUDIT
   - SEMANTIC_QA
3. Top-K Truncation Resolution:
   - Person A with 20 documents is fully retrieved with total sum without truncation.
4. FastAPI Endpoint /api/chat/query
"""

import unittest
import math
from fastapi.testclient import TestClient

from src.rag_engine import (
    EmbeddingService,
    FinancialRAGChatbot,
    QueryIntent,
    EMBEDDING_DIM
)
from src.app import app


class TestRAGChatbot(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.embed_service = EmbeddingService(dimension=EMBEDDING_DIM)
        cls.chatbot = FinancialRAGChatbot(embedding_service=cls.embed_service)
        cls.client = TestClient(app)

    def test_01_embedding_service_dimensions_and_norm(self):
        """Verify embedding dimension is 1024 and vector is L2-normalized."""
        text = "ใบเสร็จรับเงิน ค่าที่พักโรงแรมเพื่อการสัมมนาทางวิชาการ มหาวิทยาลัยมหิดล จำนวนเงิน 3,500.00 บาท"
        vector = self.embed_service.get_embedding(text)

        self.assertEqual(len(vector), EMBEDDING_DIM, f"Embedding must be {EMBEDDING_DIM}-dimensional")
        
        # Check L2 norm is ~1.0
        norm = math.sqrt(sum(x * x for x in vector))
        self.assertAlmostEqual(norm, 1.0, places=3, msg="Embedding vector must be L2-normalized")

        # Check semantic similarity between related vs unrelated text
        vec_related = self.embed_service.get_embedding("ใบเสร็จค่าที่พักสัมมนาวิชาการ 3500 บาท")
        vec_unrelated = self.embed_service.get_embedding("การขอย้ายสังกัดและเปลี่ยนตำแหน่งข้าราชการพลเรือน")

        sim_related = sum(a * b for a, b in zip(vector, vec_related))
        sim_unrelated = sum(a * b for a, b in zip(vector, vec_unrelated))

        self.assertGreater(sim_related, sim_unrelated, "Related financial texts should have higher cosine similarity")

    def test_02_intent_classification_and_routing(self):
        """Verify hybrid query router categorizes queries into the correct strategy."""
        # 1. Aggregation / Listing -> Exhaustive Aggregation
        q1 = "สรุปรายการเบิกจ่ายทั้งหมดของ นายสมชาย มั่นคง มีกี่ฉบับ รวมเป็นเงินเท่าไหร่"
        intent1, meta1 = self.chatbot.classify_query_intent(q1)
        self.assertEqual(intent1, QueryIntent.EXHAUSTIVE_AGGREGATION)
        self.assertIn("สมชาย มั่นคง", meta1.get("person_name", ""))

        # 2. Case Dossier Tracking -> Case Dossier Lookup
        q2 = "ขอเอกสารทั้งหมดที่อ้างอิงถึงบันทึกข้อความ อว 78.101/20271"
        intent2, meta2 = self.chatbot.classify_query_intent(q2)
        self.assertEqual(intent2, QueryIntent.CASE_DOSSIER_LOOKUP)
        self.assertEqual(meta2.get("case_identifier"), "อว 78.101/20271")

        # 3. Reconciliation Audit -> Reconciliation Audit
        q3 = "มีเอกสารใบไหนบ้างที่ยอดรวมไม่ตรงกับรายการย่อย math reconciliation ไม่ผ่าน"
        intent3, _ = self.chatbot.classify_query_intent(q3)
        self.assertEqual(intent3, QueryIntent.RECONCILIATION_AUDIT)

        # 4. Semantic QA -> Semantic QA
        q4 = "โครงการสัมมนา AI จัดขึ้นที่ไหนและมีเป้าหมายเพื่ออะไร"
        intent4, _ = self.chatbot.classify_query_intent(q4)
        self.assertEqual(intent4, QueryIntent.SEMANTIC_QA)

    def test_03_top_k_truncation_bypass_20_documents(self):
        """
        CRITICAL TEST CASE: The Top-K Truncation Problem
        Scenario:
        Person A ('นายสมชาย มั่นคง') has 20 expense documents.
        A naive vector search with top_k=5 would only retrieve 5 documents (missing 15 docs and wrong total).
        With our Hybrid Router:
        The query 'สรุปรายการเบิกจ่ายทั้งหมดของ นายสมชาย มั่นคง มีกี่ฉบับ รวมเป็นเงินเท่าไหร่'
        bypasses top_k truncation, queries all 20 documents, returns accurate count and total sum.
        """
        # Inject 20 mock documents for 'นายสมชาย มั่นคง' into memory store
        # 20 receipts of 750.00 THB each = total 15,000.00 THB
        mock_docs = []
        for i in range(1, 21):
            mock_docs.append({
                "id": f"mock-{1000 + i}",
                "filename": f"receipt_somchai_{i:02d}.pdf",
                "document_type": "general_receipt",
                "document_type_name": "ใบเสร็จรับเงินทั่วไป",
                "doc_no": f"REC-SC-2026-{i:03d}",
                "doc_date": f"2026-09-{min(i, 28):02d}",
                "requester": "นายสมชาย มั่นคง",
                "total_amount": 750.00,
                "embed_text": f"ใบเสร็จรับเงิน เลขที่ REC-SC-2026-{i:03d} ผู้ขอเบิก: นายสมชาย มั่นคง จำนวนเงิน: 750.00 บาท",
                "reconciliation_status": "passed",
                "reconciliation_details": "ยอดเงินตรงกับรายการย่อย",
                "principle_doc_no": "อว 78.101/20271"
            })

        # Save existing fallback docs and swap with mock docs for deterministic test
        original_fallback = self.chatbot.mock_corpus
        self.chatbot.mock_corpus = mock_docs

        try:
            # Query with top_k=5 (default)
            query = "สรุปรายการเบิกจ่ายทั้งหมดของ นายสมชาย มั่นคง มีกี่ฉบับ รวมเป็นเงินเท่าไหร่"
            result = self.chatbot.execute_rag(query, top_k=5)

            # Assertions proving Top-K Truncation is solved
            self.assertTrue(result.get("success"), "RAG query should succeed")
            self.assertEqual(result.get("route_taken"), "exhaustive_sql_aggregation")
            self.assertTrue(result.get("bypass_top_k_truncation"), "Top-k truncation must be flagged as bypassed")
            
            # Must return ALL 20 documents, NOT just 5!
            retrieved_docs = result.get("documents", [])
            self.assertEqual(len(retrieved_docs), 20, f"Expected all 20 documents to be returned, got {len(retrieved_docs)}")
            
            stats = result.get("statistics", {})
            self.assertEqual(stats.get("total_documents_count"), 20, "Document count must be exactly 20")
            self.assertAlmostEqual(stats.get("grand_total_amount"), 15000.00, places=2, msg="Grand total must be 15,000.00 THB (20 x 750)")
            
            # Check answer mentions 20 ฉบับ and 15,000.00 บาท
            answer = result.get("answer", "")
            self.assertIn("20 ฉบับ", answer)
            self.assertIn("15,000.00 บาท", answer)
            self.assertIn("สมชาย มั่นคง", answer)

        finally:
            self.chatbot.mock_corpus = original_fallback

    def test_04_case_dossier_lookup(self):
        """Verify case dossier roll-up returns all child documents linked to a principle approval number."""
        query = "ขอเอกสารทั้งหมดที่อ้างอิงถึงหนังสือขออนุมัติ อว 78.101/20271"
        result = self.chatbot.execute_rag(query, top_k=5)

        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("route_taken"), "case_dossier_lookup")
        self.assertGreaterEqual(len(result.get("documents", [])), 1)
        self.assertIn("อว 78.101/20271", result.get("answer", ""))

    def test_05_reconciliation_audit(self):
        """Verify reconciliation audit identifies documents with unbalanced math."""
        query = "ตรวจสอบเอกสารที่มีความคลาดเคลื่อนทางบัญชี ยอดรวมไม่ตรงกับรายการย่อย"
        result = self.chatbot.execute_rag(query, top_k=5)

        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("route_taken"), "reconciliation_audit")
        self.assertIn("ความคลาดเคลื่อน", result.get("answer", ""))

    def test_06_fastapi_chat_query_endpoint(self):
        """Verify POST /api/chat/query endpoint returns valid JSON response."""
        response = self.client.post(
            "/api/chat/query",
            json={"query": "ยอดรวมการเบิกจ่ายโครงการสัมมนา อว 78.101/20271 มีเท่าไหร่", "top_k": 5}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data.get("success"))
        self.assertIn("answer", data)
        self.assertIn("route_taken", data)
        self.assertIn("documents", data)

    def test_07_binary_decision_mode_all_vs_top_k(self):
        """Verify Binary Decision Intention Agent assigns 'ALL' vs 'TOP_K' correctly."""
        # 1. Total/Sum/Count Query -> Mode: ALL
        q_all = "เบิกจ่ายทั้งหมดของ นายสมชาย มั่นคง รวมเป็นเงินกี่บาท"
        intent_all, meta_all = self.chatbot.classify_query_intent(q_all)
        self.assertEqual(meta_all.get("mode"), "ALL")
        self.assertEqual(meta_all.get("scope"), "ALL")

        res_all = self.chatbot.execute_rag(q_all, top_k=5)
        self.assertEqual(res_all.get("mode"), "ALL")
        self.assertTrue(res_all.get("bypass_top_k_truncation"))

        # 2. Conceptual / Policy / Semantic Query -> Mode: TOP_K
        q_topk = "โครงการสัมมนาจัดขึ้นเพื่อวัตถุประสงค์อะไร"
        intent_topk, meta_topk = self.chatbot.classify_query_intent(q_topk)
        self.assertEqual(meta_topk.get("mode"), "TOP_K")
        self.assertEqual(meta_topk.get("scope"), "TOP_K")

        res_topk = self.chatbot.execute_rag(q_topk, top_k=5)
        self.assertEqual(res_topk.get("mode"), "TOP_K")


if __name__ == "__main__":
    unittest.main()

