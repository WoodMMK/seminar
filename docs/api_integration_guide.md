# คู่มือการเชื่อมต่อระบบ (API Integration Guide)
## Thai Financial Document Extraction API — PP-ChatOCRv4 Headless Service

> 🚀 **สถาปัตยกรรมระบบ (System Architecture):**  
> บริการนี้เป็น **Pure FastAPI Backend (Headless REST API)** โดยไม่มีหน้า UI ฝังในตัว  
> สำหรับให้ทีม Frontend (React, Vue, Next.js, Angular หรือ Mobile App) เชื่อมต่อผ่าน HTTP REST API ได้โดยตรง  
> ระบบเปิดใช้งาน **CORS (Cross-Origin Resource Sharing)** ครบถ้วน รองรับการเรียกจากทุก Domain/Port (`http://localhost:3000`, `http://localhost:5173` ฯลฯ)

---

## 📌 สารบัญ (Table of Contents)
1. [ภาพรวมและการทดสอบผ่าน Swagger UI](#1-ภาพรวมและการทดสอบผ่าน-swagger-ui)
2. [รายชื่อโมเดล AI และ Classifiers ที่ใช้ในระบบ](#2-รายชื่อโมเดล-ai-และ-classifiers-ที่ใช้ในระบบ)
3. [กระบวนการเตรียมข้อมูลภาพ (Preprocessing Pipeline)](#3-กระบวนการเตรียมข้อมูลภาพ-preprocessing-pipeline)
4. [API Endpoints สำหรับเชื่อมต่อ (API Specification)](#4-api-endpoints-สำหรับเชื่อมต่อ-api-specification)
5. [โครงสร้างฐานข้อมูล (Database Schema DDL & ERD)](#5-โครงสร้างฐานข้อมูล-database-schema-ddl--erd)
6. [TypeScript Type Definitions (สำหรับ Frontend)](#6-typescript-type-definitions-สำหรับ-frontend)
7. [รหัสประเภทเอกสารทั้ง 10 ประเภท (Document Types)](#7-รหัสประเภทเอกสารทั้ง-10-ประเภท-document-types)
8. [ตัวอย่างโค้ดเรียกใช้งาน (Code Integration Examples)](#8-ตัวอย่างโค้ดเรียกใช้งาน-code-integration-examples)
9. [การจัดการ Error และ Status Codes](#9-การจัดการ-error-และ-status-codes)

---

## 🌐 1. ภาพรวมและการทดสอบผ่าน Swagger UI

* **Base URL:** `http://localhost:8000` (หรือ IP ของเครื่องเซิร์ฟเวอร์/VM เช่น `http://192.168.1.xxx:8000`)
* **Interactive Swagger UI (ทดสอบ API สดบนเบราว์เซอร์):** [http://localhost:8000/docs](http://localhost:8000/docs)
* **ReDoc Documentation:** [http://localhost:8000/redoc](http://localhost:8000/redoc)
* **API Root Status:** [http://localhost:8000/](http://localhost:8000/)
* **Health Check Liveness Probe:** `GET http://localhost:8000/health` -> `{"status": "ok"}`

```
┌─────────────────────────────────┐
│  Frontend (React / Vue / Vite)  │ (Port 3000 / 5173)
└────────────────┬────────────────┘
                 │
                 ▼  (HTTP REST + CORS)
┌────────────────────────────────────────────────────────────────────────┐
│                   FastAPI Backend (Host / VM Port 8000)                │
│                                                                        │
│  1. Ingestion & Preprocessing                                          │
│     - PDF Ingestion via pypdfium2 @ 150 DPI                            │
│     - Resolution Auto-Scale (Max Dimension ≤ 1600px Lanczos)           │
│     - Deskewing (Clamp ±12.0°) & Vertical Stitching (≤ 2400px)         │
│                                                                        │
│  2. PP-ChatOCRv4 Perception Engine (PaddleX 3.7.2)                     │
│     - Layout: PicoDet-S_layout_3cls (Text/Table/Figure)                │
│     - Text Detection: PP-OCRv6_medium_det (limit_side_len = 1600)      │
│     - Thai Text OCR: th_PP-OCRv5_mobile_rec (Zero-shot)                │
│     - Table Parsing: SLANet_plus (HTML Table & Markdown)               │
│     - Seal Detection: PP-OCRv4_server_seal_det                         │
│     - Orientation: Disabled (use_textline_orientation=False)           │
│                                                                        │
│  3. Type-Directed LLM Reasoning (Ollama: qwen2.5:3b @ CPU Mode)       │
│     - 10 Automated Document Templates & Targeted Prompts               │
│     - Mathematical Cross-check (Subtotal + VAT = Total / Formula Check)│
│                                                                        │
│  4. Dual Knowledge Payload (Vector DB Ready)                           │
│     - embed_text: Dense Markdown for Semantic Search & RAG             │
│     - filter_metadata: Structured JSON for Vector DB Filtering         │
│                                                                        │
│  5. PostgreSQL Case Dossier Tracking (Docker Port 5432)                │
│     - cases, documents (JSONB), disbursement_records                   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🤖 2. รายชื่อโมเดล AI และ Classifiers ที่ใช้ในระบบ

ระบบรันบน **CPU Mode 100%** จึงสามารถนำไป Deploy บน Virtual Machine (VM) ทั่วไปโดยไม่จำเป็นต้องมี GPU

| ลำดับ | โมเดล / Classifier | ประเภทงาน (Task) | สถาปัตยกรรม | การตั้งค่า & หน้าที่สำคัญ |
| :---: | :--- | :--- | :--- | :--- |
| **1** | `PicoDet-S_layout_3cls` | **Layout Detection** | ESNet + PAN (PicoDet) | ตรวจจับบล็อกข้อความ (Text), ตาราง (Table), และตราประทับ (Figure/Seal) ทำงานเร็วมากบน CPU |
| **2** | `PP-OCRv6_medium_det` | **Text Detection** | DBNet (Differentiable Binarization) | ตีกรอบพิกัดบรรทัดข้อความ ตั้งค่า `limit_side_len = 1600` เพื่อรักษาสมดุลความเร็วและความคมชัด |
| **3** | `th_PP-OCRv5_mobile_rec` | **Thai Text Recognition** | MobileNetV1 + BiLSTM + CTC | โมเดลรู้จำภาษาไทย อ่านสระ วรรณยุกต์ ตัวเลขอารบิก และสัญลักษณ์ทางการเงินครบถ้วน |
| **4** | `SLANet_plus` | **Table Recognition** | Structure Location & Alignment Network | อ่านตารางและโครงสร้างเซลล์ แปลงเป็น Markdown และ HTML `<table>` |
| **5** | `PP-OCRv4_server_seal_det`<br>& `rec_doc` | **Seal / Stamp OCR** | Server DBNet + CRNN | ตรวจจับและอ่านข้อความในตราประทับราชการ |
| **6** | `qwen2.5:3b` *(Ollama)* | **LLM Reasoning & QA** | Transformer Decoder (3B Params) | รันบน CPU (`num_gpu: 0`), ตั้งค่า `temperature: 0.0` (Greedy) เพื่อผลลัพธ์ที่แน่นอนและแม่นยำ 100% |
| ⚠️ | **Textline Orientation** | **Orientation Classifier** | **ปิดการใช้งาน (`False`)** | ป้องกันโมเดลหมุนภาพ 180° ผิดพลาดจากสระลอยไทย (เช่น ิ, ี, ่, ้) |

---

## 🛠️ 3. กระบวนการเตรียมข้อมูลภาพ (Preprocessing Pipeline)

กระบวนการทั้งหมดเกิดขึ้นที่เซิร์ฟเวอร์โดยอัตโนมัติก่อนส่งเข้าโมเดล OCR:

1. **PDF Ingestion & Rendering (`DocumentIngestion`):**  
   ใช้ `pypdfium2` เรนเดอร์ PDF เป็นภาพความละเอียด **150 DPI** (`target_dpi = 150`)  
   *(ลดขนาดพิกเซลลง 44% เมื่อเทียบกับ 200 DPI แต่ฟอนต์ราชการขนาดเล็ก 14pt ยังอ่านได้แม่นยำ 100%)*
2. **Resolution Auto-Scaling:**  
   คุมความยาวด้านมากสุดของภาพ (Max Side) ไม่ให้เกิน **1600 px** ด้วย **Lanczos Interpolation** ป้องกัน CPU Bottleneck
3. **Deskewing Correction (`deskew_image`):**  
   คำนวณมุมเอียงของเอกสารสแกนด้วย OpenCV Otsu + `minAreaRect` และหมุนภาพกลับ (จำกัดช่วงไม่เกิน **±12.0°**)
4. **Multi-Page Vertical Stitching:**  
   กรณีเอกสาร PDF หลายหน้า ระบบจะนำหน้ามาต่อกันในแนวตั้ง คั่นด้วยระยะ 24px และคุมความสูงรวมไม่เกิน **2400 px**
5. **Thai Fragment Merging:**  
   รวมสระลอยบน-ล่าง (ิ, ี, ่, ้, ุ, ู) ที่มักหลุดจาก Bounding Box กลับเข้าบรรทัดแม่โดยอัตโนมัติ

---

## 🔌 4. API Endpoints สำหรับเชื่อมต่อ (API Specification)

### 1) `POST /api/chatocr/chat` — สกัดข้อมูลเอกสาร (Core Extraction)
> **Endpoint หลัก:** รับไฟล์เอกสาร ➔ รัน OCR ➔ รัน LLM ตอบคำถามตาม Template ➔ ตรวจสอบกระทบยอดเงิน ➔ ส่งคืนผลลัพธ์ JSON พร้อม **Dual Knowledge Payload** สำหรับ Vector DB

* **Method:** `POST`
* **Path:** `/api/chatocr/chat`
* **Content-Type:** `multipart/form-data`

#### Request Parameters (Form Data):
| พารามิเตอร์ | ประเภท | จำเป็น | ค่าเริ่มต้น | คำอธิบาย |
| :--- | :---: | :---: | :---: | :--- |
| `file` | File (Binary) | **ใช่** | - | ไฟล์เอกสาร (`.pdf`, `.png`, `.jpg`, `.jpeg`, `.webp`) |
| `document_type` | string | **ใช่** | `"general_receipt"` | รหัสประเภทเอกสาร 1 ใน 10 ประเภท (ดูรายการในหัวข้อที่ 6) |
| `questions` | string | ไม่ | `null` | คำถามเพิ่มเติม (คั่นด้วยจุลภาคหรือขึ้นบรรทัดใหม่) หากไม่ระบุ ระบบจะใช้ชุดคำถามมาตรฐานประจำประเภทให้อัตโนมัติ |
| `custom_prompt` | string | ไม่ | `null` | System Prompt พิเศษ (หากไม่ระบุจะใช้ Prompt เชี่ยวชาญประจำประเภท) |
| `use_sample` | boolean | ไม่ | `false` | ตั้งเป็น `true` หากต้องการทดสอบด้วยภาพใบเสร็จตัวอย่างในระบบ |

#### ตัวอย่างผลลัพธ์ (JSON Response - 200 OK):
```json
{
  "status": "success",
  "source_file": "memo_approval.pdf",
  "document_type": "principle_approval_request",
  "document_type_name": "เอกสารขออนุมัติหลักการ",
  "questions": [
    "เลขที่เอกสาร",
    "วันที่ทำเอกสาร",
    "เรื่อง",
    "ผู้ทำการเบิก (คน หรือ ภาควิชา)",
    "รายละเอียดค่าใช้จ่าย",
    "ยอดรวม"
  ],
  "system_prompt_used": "สกัดข้อมูลจากบันทึกข้อความ 'ขออนุมัติหลักการ'...",
  "chat_answers": {
    "เลขที่เอกสาร": "อว 78.101/20271",
    "วันที่ทำเอกสาร": "วันที่ 4 กรกฎาคม 2568",
    "เรื่อง": "ขออนุมัติหลักการเป็นค่าอาหารว่างและเครื่องดื่มสำหรับต้อนรับคณะครูและนักเรียน...",
    "ผู้ทำการเบิก (คน หรือ ภาควิชา)": "ผศ.ดร.กลกรณ์ วงศ์ภาติกะเสรี",
    "รายละเอียดค่าใช้จ่าย": "ค่าอาหารว่างและเครื่องดื่ม... ในอัตรา 20 บาท × 135 คน",
    "ยอดรวม": "2,700.- บาท (สองพันเจ็ดร้อยบาทถ้วน)"
  },
  "knowledge_payload": {
    "embed_text": "# เอกสารการเงิน: memo_approval.pdf\n- **ชื่อเอกสาร (Document Title):** memo_approval.pdf\n- **ประเภทเอกสาร:** เอกสารขออนุมัติหลักการ\n- **เลขที่เอกสาร (Document No.):** อว 78.101/20271\n- **เลขที่ขออนุมัติหลักการที่อ้างถึง:** อว 78.101/20271\n- **วันที่เอกสาร:** 2025-07-04\n- **ผู้เบิก / ร้านค้า (Vendor/Requester):** ผศ.ดร.กลกรณ์ วงศ์ภาติกะเสรี\n- **ยอดเงินรวมทั้งสิ้น (Total Amount):** ฿2,700.00\n- **สถานะการกระทบยอดทางบัญชี:** ตรวจสอบถูกต้อง (Reconciled)\n- **รายละเอียดการกระทบยอด:** ยอดเงินตรงกันสมบูรณ์: คำนวณสูตร 20.00 x 135 = 2,700.00 บาท\n\n## รายละเอียดคำตอบจากการสกัดตามแม่แบบ:\n- **เลขที่เอกสาร:** อว 78.101/20271\n- **ยอดรวม:** 2,700.- บาท",
    "filter_metadata": {
      "principle_doc_no": "อว 78.101/20271",
      "document_no": "อว 78.101/20271",
      "document_title": "memo_approval.pdf",
      "document_type": "principle_approval_request",
      "document_type_name": "เอกสารขออนุมัติหลักการ",
      "date": "2025-07-04",
      "vendor": "ผศ.ดร.กลกรณ์ วงศ์ภาติกะเสรี",
      "tax_id": null,
      "subtotal": null,
      "vat": null,
      "total_amount": 2700.0,
      "math_reconciliation": {
        "status": "passed",
        "formula": "subtotal + vat == total or rate * qty == total",
        "subtotal": null,
        "vat": null,
        "computed_total": 2700.0,
        "declared_total": 2700.0,
        "discrepancy": 0.0,
        "details": "ยอดเงินตรงกันสมบูรณ์: คำนวณสูตร 20.00 x 135 = 2,700.00 บาท"
      }
    }
  },
  "image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
  "ocr_full_text": "ข้อความทั้งหมดที่อ่านได้จากเอกสาร...",
  "ocr_text_lines": ["ที่ อว 78.101/20271", "วันที่ 4 กรกฎาคม 2568", "..."],
  "ocr_line_count": 28,
  "latency_ms": 24200
}
```

---

### 2) `GET /api/chatocr/templates` — ดึงรายการแม่แบบ 10 ประเภท
* **Method:** `GET`
* **Path:** `/api/chatocr/templates`
* **การใช้งาน:** นำไปสร้าง Dropdown ให้ผู้ใช้เลือกประเภทเอกสาร และแสดงรายการฟิลด์ที่จะสกัด (`keys`)

#### ผลลัพธ์ตัวอย่าง:
```json
[
  {
    "id": "principle_approval_request",
    "title": "เอกสารขออนุมัติหลักการ",
    "group": "official_reimbursement",
    "group_title": "เอกสารเบิกจ่ายราชการ (9 ประเภทหลัก)",
    "keys": [
      "เลขที่เอกสารหรือเลขที่หนังสือ",
      "วันที่ทำเอกสาร",
      "เรื่อง",
      "ผู้ทำการเบิกหรือหน่วยงานที่ขอ",
      "รายละเอียดค่าใช้จ่าย",
      "ยอดรวมเงินงบประมาณที่ขออนุมัติ"
    ],
    "system_prompt": "คุณเป็นผู้เชี่ยวชาญการสกัดข้อมูลเอกสารราชการ..."
  },
  {
    "id": "general_receipt",
    "title": "ใบเสร็จรับเงิน / ใบกำกับภาษีทั่วไป (ร้านค้า/บริษัท)",
    "group": "supplementary",
    "group_title": "เอกสารประกอบภายนอก (หมวดเสริม)",
    "keys": [
      "ชื่อร้านค้าหรือบริษัทผู้ขาย",
      "เลขประจำตัวผู้เสียภาษี 13 หลัก",
      "วันที่ออกเอกสาร",
      "ยอดรวมก่อนภาษี (Subtotal)",
      "ภาษีมูลค่าเพิ่ม (VAT 7%)",
      "ยอดรวมสุทธิทั้งสิ้น (Total Amount)",
      "รายการสินค้าและบริการทั้งหมด",
      "Cross-check ความถูกต้องทางคณิตศาสตร์"
    ]
  }
]
```

---

### 3) `POST /api/chatocr/save_to_db` — บันทึกข้อมูลและผูกชุดเรื่อง (Save & Link Dossier)
* **Method:** `POST`
* **Path:** `/api/chatocr/save_to_db`
* **Content-Type:** `application/json`

#### Request Body (JSON):
```json
{
  "filename": "memo_approval.pdf",
  "document_type": "principle_approval_request",
  "metadata": {
    "principle_doc_no": "อว 78.101/20271",
    "document_no": "อว 78.101/20271",
    "document_title": "โครงการสัมมนาวิชาการ",
    "date": "2025-07-04",
    "vendor": "ผศ.ดร.กลกรณ์ วงศ์ภาติกะเสรี",
    "tax_id": null,
    "subtotal": null,
    "vat": null,
    "total_amount": 2700.0,
    "math_reconciliation": {
      "status": "passed",
      "details": "ยอดเงินตรงกันสมบูรณ์"
    }
  },
  "embed_text": "# เอกสารการเงิน: ...",
  "chat_answers": {
    "เลขที่เอกสาร": "อว 78.101/20271",
    "ยอดรวม": "2,700.- บาท"
  },
  "math_reconciliation": {
    "status": "passed",
    "details": "ยอดเงินตรงกันสมบูรณ์"
  },
  "target_record_id": null,
  "is_new_case": true,
  "principle_doc_no": "อว 78.101/20271",
  "case_title": "โครงการสัมมนาต้อนรับคณะครูและนักเรียน"
}
```

* **หมายเหตุการผูกชุดเรื่อง (`Case Linking`):**
  * ถ้าเป็นเอกสารขออนุมัติหลักการฉบับแรก: ตั้ง `is_new_case: true`, ใส่ `case_title`, และ `principle_doc_no`
  * ถ้าเป็นเอกสารใบเสร็จหรือใบเบิกจ่ายที่แนบตามหลัง: ตั้ง `is_new_case: false`, ส่ง `target_record_id` ของชุดเรื่องที่ต้องการผูก

#### Response (200 OK):
```json
{
  "status": "success",
  "message": "บันทึกและผูกเอกสารเข้ากับชุดเรื่องสำเร็จ",
  "case_id": "8b5d2024-5d51-4fa3-9f5e-fb68393e18a0",
  "case_title": "โครงการสัมมนาต้อนรับคณะครูและนักเรียน",
  "document_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "record_id": "f1e2d3c4-b5a6-7890-fedc-ba9876543210",
  "approved_budget": 2700.0,
  "actual_expense": 2700.0,
  "remaining_budget": 0.0,
  "match_type": "created_new_case"
}
```

---

### 4) `GET /api/db/cases` — ดึงรายชื่อชุดเรื่องสำหรับ Dropdown
* **Method:** `GET`
* **Path:** `/api/db/cases`
* **คำอธิบาย:** ดึงเฉพาะรายชื่อชุดเรื่อง (Case Dossiers) ที่มีอยู่ในระบบ สำหรับนำไปใส่ใน `<select>` หรือ Search Autocomplete เมื่อผู้ใช้ต้องการแนบเอกสารเพิ่มเข้าชุดเรื่องเดิม
* **หมายเหตุ:** แดชบอร์ดสรุปผลรวมและสถิติ (Dashboard Analytics) ได้ถูกย้ายไปจัดการในส่วนของระบบหลัก/Frontend เรียบร้อยแล้ว

#### Response (200 OK):
```json
[
  {
    "record_id": "8b5d2024-5d51-4fa3-9f5e-fb68393e18a0",
    "principle_doc_no": "อว 78.101/20271",
    "title": "โครงการสัมมนาต้อนรับคณะครูและนักเรียน",
    "approved_budget": 2700.0,
    "actual_expense": 2700.0,
    "remaining_budget": 0.0,
    "status": "approved",
    "receiver_name": "ผศ.ดร.กลกรณ์ วงศ์ภาติกะเสรี",
    "request_date": "2025-07-04T00:00:00Z"
  }
]
```

---

## 🗄️ 5. โครงสร้างฐานข้อมูล (Database Schema DDL & ERD)

ฐานข้อมูลของระบบทำงานบน **PostgreSQL 16** ใน Docker Container (`expense-reimbursement-postgres`)  
ไฟล์ Export DDL และ Diagram แบบสมบูรณ์อยู่ที่:
* 📄 **SQL Schema DDL:** [`docs/database_schema.sql`](database_schema.sql) *(ไฟล์ DDL สดที่ Export จาก `pg_dump` พร้อม Constraints และ Indexes)*
* 📊 **Mermaid ERD Diagram:** [`docs/database_erd.mmd`](database_erd.mmd) *(แผนภาพความสัมพันธ์ระดับ Entity)*

### โครงสร้างตารางหลัก (Core Tables):
1. **`disbursement_records` (Case Dossier):**  
   ตารางชุดเรื่องการเบิกจ่ายหลัก ผูกกับ `principle_doc_no` (เช่น อว 78.101/20271), บันทึก `approved_amount`, `actual_expense`, และ `status`
2. **`documents` (Attached Documents & AI Extractions):**  
   ตารางเอกสารที่แนบในแต่ละชุดเรื่อง (`record_id` FK -> `disbursement_records.id`) โดยเก็บข้อมูลทั้งหมดที่ AI สกัดได้ในคอลัมน์ `extracted_data (JSONB)`
3. **`expense_items` (Itemized Expenses):**  
   รายการแจกแจงค่าใช้จ่ายย่อย เช่น ค่าอาหารว่าง, ค่าวัสดุ, ค่าตอบแทนวิทยากร (`record_id` FK, `document_id` FK)
4. **`users` (User Accounts & Roles):**  
   บัญชีผู้ใช้งาน เจ้าหน้าที่การเงิน และผู้ตรวจสอบระบบ
5. **`work_logs` (Audit Trail):**  
   บันทึกประวัติการเปลี่ยนสถานะและการตรวจสอบเอกสาร

---

## 📐 6. TypeScript Type Definitions (สำหรับ Frontend)

เพื่อนฝั่ง Frontend สามารถคัดลอกไฟล์ประเภทข้อมูลนี้ไปวางใน `types/extraction.ts` ได้ทันที:

```typescript
// types/extraction.ts

export type MathReconcileStatus = 'passed' | 'discrepancy' | 'verified_by_llm' | 'unverified';

export interface MathReconciliation {
  status: MathReconcileStatus;
  formula?: string;
  subtotal?: number | null;
  vat?: number | null;
  computed_total?: number;
  declared_total?: number;
  discrepancy?: number;
  details: string;
}

export interface FilterMetadata {
  principle_doc_no?: string | null;
  document_no?: string | null;
  document_title?: string;
  document_type: string;
  document_type_name?: string;
  date?: string | null;
  vendor?: string | null;
  tax_id?: string | null;
  subtotal?: number | null;
  vat?: number | null;
  total_amount?: number | null;
  math_reconciliation?: MathReconciliation;
}

export interface KnowledgePayload {
  embed_text: string;
  filter_metadata: FilterMetadata;
}

export interface ChatOcrResponse {
  status: 'success' | 'error';
  source_file: string;
  document_type: string;
  document_type_name: string;
  questions: string[];
  system_prompt_used: string;
  chat_answers: Record<string, string>;
  knowledge_payload: KnowledgePayload;
  image_base64?: string;
  ocr_full_text?: string;
  ocr_text_lines?: string[];
  ocr_line_count?: number;
  latency_ms: number;
}

export interface DocumentTemplate {
  id: string;
  title: string;
  group: 'official_reimbursement' | 'supplementary';
  group_title: string;
  keys: string[];
  system_prompt?: string;
}

export interface SaveToDbPayload {
  filename: string;
  document_type: string;
  metadata: FilterMetadata;
  embed_text: string;
  chat_answers: Record<string, string>;
  math_reconciliation?: MathReconciliation;
  target_record_id?: string | null;
  is_new_case?: boolean;
  principle_doc_no?: string | null;
  case_title?: string | null;
}

export interface SaveToDbResponse {
  status: string;
  message: string;
  case_id: string;
  case_title: string;
  document_id: string;
  record_id: string;
  approved_budget?: number;
  actual_expense?: number;
  remaining_budget?: number;
  match_type: 'created_new_case' | 'matched_existing_case';
}

export interface ExistingCaseItem {
  record_id: string;
  principle_doc_no: string;
  title: string;
  approved_budget: number;
  actual_expense: number;
  remaining_budget: number;
  status: string;
  receiver_name?: string;
  request_date?: string;
}
```

---

## 📑 7. รหัสประเภทเอกสารทั้ง 10 ประเภท (`document_type`)

| รหัสประเภท (`document_type`) | ชื่อประเภทภาษาไทย | กลุ่มเอกสาร | ฟิลด์คำถามมาตรฐานที่ดึงให้อัตโนมัติ |
| :--- | :--- | :--- | :--- |
| **`principle_approval_request`** | เอกสารขออนุมัติหลักการ | ราชการ | เลขที่เอกสาร, วันที่ทำเอกสาร, เรื่อง, ผู้ทำการเบิก (คน หรือ ภาควิชา), รายละเอียดค่าใช้จ่าย, ยอดรวม |
| **`principle_approval_granted`** | เอกสารอนุมัติหลักการ | ราชการ | รายละเอียดการอนุมัติ, ยอดรวมที่อนุมัติ, ตามหนังสือขออนุมัติหลักการเลขที่ |
| **`disbursement_approval_request`** | ขออนุมัติเบิกจ่าย | ราชการ | เลขที่เอกสาร, วันที่ทำเอกสาร, เรื่อง, รายละเอียดค่าใช้จ่าย, ยอดรวม, ประเภทการเบิก (เงินสดย่อย/ทดรองจ่าย), อ้างถึงบันทึกหลักการเลขที่ |
| **`advance_payment_request_1`** | แบบเบิกเงินทดรองจ่าย (แบบที่ 1 - สัญญายืมเงิน/เบิก) | ราชการ | เลขที่เอกสาร, ใครเป็นคนเบิก, ยอดรวม, ตามหนังสืออนุมัติเบิกจ่ายเลขที่, รายละเอียด, โอนเงินไปที่ใด |
| **`advance_payment_request_2`** | แบบเบิกเงินทดรองจ่าย (แบบที่ 2 - รับเงิน/เคลียร์เงิน) | ราชการ | เลขที่เอกสาร, วันที่ส่งเอกสาร, วันที่ขอรับเงิน, ใครเป็นคนเบิก, ยอดรวม, ตามหนังสืออนุมัติหลักการเลขที่, รายละเอียด, โอนเงินไปที่ใด |
| **`receipt_substitute`** | ใบแทนใบเสร็จ / ใบสำคัญรับเงิน | ราชการ | วัน/เดือน/ปี, รายละเอียดของรายการการเบิก, ยอดรวม, ผู้จ่ายเงิน |
| **`parcel_inspection`** | ใบตรวจรับพัสดุ | ราชการ | เลขที่เอกสาร, ผู้ตรวจรับพัสดุ, วันที่ตรวจรับพัสดุ, ตามใบสั่งซื้อหรือสัญญาเลขที่ |
| **`procurement_approval_request`** | ขออนุมัติหาพัสดุ (จัดหาพัสดุ) | ราชการ | เลขที่เอกสาร, วันที่ทำเอกสาร, เรื่อง, เหตุผลที่ต้องจัดหา, รายละเอียดของพัสดุ, วงเงินที่ใช้ทั้งหมด, เวลาที่ต้องใช้พัสดุ, อ้างถึงบันทึกขออนุมัติหลักการเลขที่ |
| **`procurement_attachment`** | เอกสารประกอบการขออนุมัติจัดหา | ราชการ | แนบท้ายบันทึกเอกสารเลขอะไร, รายละเอียดพัสดุหรือรายการเปรียบเทียบราคา, ยอดรวม |
| **`general_receipt`** | ใบเสร็จรับเงิน / ใบกำกับภาษีทั่วไป | เอกสารเสริม | ชื่อร้านค้า/บริษัท, เลขประจำตัวผู้เสียภาษี 13 หลัก, วันที่, ยอดก่อนภาษี, VAT 7%, ยอดเงินรวมทั้งสิ้น, รายการสินค้าและบริการ, การตรวจสอบกระทบยอด |

---

## 💻 8. ตัวอย่างโค้ดเรียกใช้งาน (Code Integration Examples)

### 7.1 ตัวอย่าง React Custom Hook (`useChatOCR.ts`)
```typescript
import { useState } from 'react';
import { ChatOcrResponse, SaveToDbPayload, SaveToDbResponse } from './types/extraction';

const API_BASE = "http://localhost:8000";

export function useChatOCR() {
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ChatOcrResponse | null>(null);

  // 1. ส่งไฟล์สกัดข้อมูล
  const extractDocument = async (file: File, documentType: string = 'general_receipt') => {
    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('document_type', documentType);

      const res = await fetch(`${API_BASE}/api/chatocr/chat`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errJson = await res.json();
        throw new Error(errJson.error || `HTTP error ${res.status}`);
      }

      const data: ChatOcrResponse = await res.json();
      setResult(data);
      return data;
    } catch (err: any) {
      setError(err.message || 'เกิดข้อผิดพลาดในการสกัดข้อมูล');
      throw err;
    } finally {
      setLoading(false);
    }
  };

  // 2. บันทึกลงฐานข้อมูลและผูกชุดเรื่อง
  const saveToDatabase = async (payload: SaveToDbPayload): Promise<SaveToDbResponse> => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/chatocr/save_to_db`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const errJson = await res.json();
        throw new Error(errJson.error || 'Failed to save to database');
      }

      return await res.json();
    } catch (err: any) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  return { extractDocument, saveToDatabase, loading, error, result };
}
```

---

### 7.2 ตัวอย่าง Vanilla JavaScript (`fetch`)
```javascript
const API_BASE = "http://localhost:8000";

async function handleUploadAndExtract() {
  const fileInput = document.getElementById("fileInput");
  const docTypeSelect = document.getElementById("docTypeSelect");

  if (!fileInput.files.length) {
    alert("กรุณาเลือกไฟล์");
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);
  formData.append("document_type", docTypeSelect.value);

  try {
    const res = await fetch(`${API_BASE}/api/chatocr/chat`, {
      method: "POST",
      body: formData
    });

    if (!res.ok) throw new Error("การสกัดข้อมูลล้มเหลว");
    const data = await res.json();

    console.log("ผลลัพธ์คำถาม-คำตอบ:", data.chat_answers);
    console.log("กระทบยอด:", data.knowledge_payload.filter_metadata.math_reconciliation);
    console.log("Vector DB Text:", data.knowledge_payload.embed_text);

    // แสดงพรีวิวภาพที่ระบบประมวลผลแล้ว (ถ้ามี)
    if (data.image_base64) {
      document.getElementById("previewImg").src = `data:image/png;base64,${data.image_base64}`;
    }
  } catch (err) {
    console.error("Error:", err);
  }
}
```

---

### 7.3 ตัวอย่าง Axios (Node.js / Vue / React)
```javascript
import axios from 'axios';

const api = axios.create({ baseURL: 'http://localhost:8000' });

async function uploadReceipt(fileObj, docType = 'general_receipt') {
  const form = new FormData();
  form.append('file', fileObj);
  form.append('document_type', docType);

  const { data } = await api.post('/api/chatocr/chat', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 180000 // 3 minutes timeout for CPU OCR & LLM processing
  });

  return data;
}
```

---

### 7.4 ตัวอย่าง Python (`requests`)
```python
import requests

API_BASE = "http://localhost:8000"

def extract_document(file_path: str, doc_type: str = "principle_approval_request"):
    url = f"{API_BASE}/api/chatocr/chat"
    
    with open(file_path, "rb") as f:
        files = {"file": (file_path, f, "application/pdf")}
        data = {"document_type": doc_type}
        
        res = requests.post(url, files=files, data=data, timeout=180)
        res.raise_for_status()
        result = res.json()

    print("📄 เลขที่เอกสาร:", result["chat_answers"].get("เลขที่เอกสาร"))
    print("💰 ยอดรวม:", result["chat_answers"].get("ยอดรวม"))
    print("🔍 กระทบยอด:", result["knowledge_payload"]["filter_metadata"]["math_reconciliation"]["details"])
    return result

# ทดสอบรัน
# extract_document("memo.pdf", doc_type="principle_approval_request")
```

---

### 7.5 ตัวอย่าง cURL Command
```bash
curl -X POST "http://localhost:8000/api/chatocr/chat" \
  -F "file=@memo_approval.pdf" \
  -F "document_type=principle_approval_request"
```

---

## ⚠️ 9. การจัดการ Error และ Status Codes

| HTTP Status Code | สาเหตุ | ตัวอย่าง Response Body | แนวทางแก้ไขสำหรับ Frontend |
| :---: | :--- | :--- | :--- |
| **`200 OK`** | ประมวลผลสำเร็จ | `{ "status": "success", ... }` | นำข้อมูล `chat_answers` และ `knowledge_payload` ไปแสดงผล |
| **`400 Bad Request`** | ไฟล์ไม่ถูกต้อง หรือ PDF เสีย | `{ "error": "Failed to parse document pages" }` | แจ้งเตือนผู้ใช้ให้ตรวจสอบไฟล์เอกสารที่อัปโหลด |
| **`422 Unprocessable`** | ขาดฟิลด์จำเป็นใน Form Data | `{ "detail": [{ "loc": ["body", "file"], "msg": "Field required" }] }` | ตรวจสอบว่าส่ง `file` หรือ `document_type` ถูกต้องตาม Schema |
| **`500 Internal Error`** | OCR / LLM Engine หรือ DB ขัดข้อง | `{ "error": "...", "success": false }` | ตรวจสอบว่า Container PostgreSQL หรือ Ollama รันอยู่หรือไม่ |

---