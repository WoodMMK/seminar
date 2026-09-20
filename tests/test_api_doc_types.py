import json
import urllib.request

def test_api_types():
    print("Testing GET /api/llm/status...")
    with urllib.request.urlopen("http://localhost:8000/api/llm/status", timeout=5) as resp:
        data = json.loads(resp.read().decode())
    
    print(f"Online: {data['ollama_online']}")
    print(f"Supported Types Count: {len(data['supported_document_types'])}")
    for dt in data['supported_document_types']:
        print(f"  - {dt['id']}: {dt['title']}")

    # 1. Test Advance Payment Request (Type 4)
    print("\nTesting POST /api/llm/extract for advance_payment_request_1...")
    payload1 = json.dumps({
        "ocr_markdown": "แบบเบิกเงินทดรองจ่าย ยง. 12/2567 ผู้เบิก ผศ.ดร.สมชาย ใจดี ยอด 12,000 บาท ตามหนังสือ อว 78.03/0892 โอนเข้าไทยพาณิชย์",
        "document_type": "advance_payment_request_1",
        "force_mock": True
    }).encode("utf-8")
    req1 = urllib.request.Request("http://localhost:8000/api/llm/extract", data=payload1, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req1, timeout=10) as resp:
        res1 = json.loads(resp.read().decode())
    
    print(f"Type 4 Result:")
    print(f"  Title: {res1['document_type_name_th']}")
    print(f"  Fields: {res1['fields']}")
    assert res1['document_type'] == "advance_payment_request_1"
    assert "transfer_destination" in res1['fields']
    assert "ref_doc_no" in res1['fields']

    # 2. Test Parcel Inspection (Type 7)
    print("\nTesting POST /api/llm/extract for parcel_inspection...")
    payload2 = json.dumps({
        "ocr_markdown": "ใบตรวจรับพัสดุ บพ. 88/2567 คณะกรรมการ รศ.ดร.สมบัติ",
        "document_type": "parcel_inspection",
        "force_mock": True
    }).encode("utf-8")
    req2 = urllib.request.Request("http://localhost:8000/api/llm/extract", data=payload2, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req2, timeout=10) as resp:
        res2 = json.loads(resp.read().decode())
    
    print(f"Type 7 Result:")
    print(f"  Title: {res2['document_type_name_th']}")
    print(f"  Fields: {res2['fields']}")
    assert res2['document_type'] == "parcel_inspection"
    assert "inspectors" in res2['fields']

    print("\n✅ API Document Type Integration Tests PASSED!")

if __name__ == "__main__":
    test_api_types()
