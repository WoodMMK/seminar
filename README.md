# Thai Financial Document Extraction REST API Backend

ระบบสกัดข้อมูลและตรวจสอบความถูกต้องของเอกสารเบิกจ่ายราชการและเอกสารการเงินภาษาไทยอัตโนมัติ (**Headless FastAPI Backend Service** ขับเคลื่อนด้วย PP-ChatOCRv4, Thai PP-OCRv5 และ Local LLM)

## Latest ResNet taxonomy and extraction routing

The extraction API accepts either the canonical extraction type or the exact
class label produced by the latest ResNet model. The authoritative mapping is
available from `GET /api/v1/document-types`.

Important mappings:

- `04PettyCash` -> `advance_payment_request_2` (แบบที่ 2 - เงินสดย่อย)
- `05AdvancePayment` -> `advance_payment_request_1` (แบบที่ 1 - เงินทดรองจ่าย)
- `07_02Inspectionform` -> `parcel_inspection_page_2`
- `10Approved_materials` -> `approved_materials`
- `10Additional_documents` -> `additional_documents` (extraction is skipped)

New extraction fields:

- `07_02Inspectionform`: เลขที่เอกสาร, ชื่อผู้ลงนาม
- `10Approved_materials`: รหัสใบขอจัดหา, ยอดรวมที่อนุมัติ,
  รายละเอียดรายการที่อนุมัติ, วันที่จัดทำเอกสาร, เรื่อง, ผู้ตรวจรับพัสดุ

Additional documents return `should_extract: false`, an empty `fields` object,
and a `skip_reason`; they are never sent to the extraction LLM.

## Full automatic pipeline

Use `POST /api/v1/classify-validate-extract` with only a PDF or image upload.
The endpoint performs:

```text
input
  -> latest 11-class ResNet (prediction, confidence, top-3)
  -> Thai OCR, processed page by page
  -> load the predicted class's positive reference OCR (`01.json`)
  -> one Qwen request for both validation and conditional extraction
```

The combined LLM result is handled as follows:

- `MATCH`: extract with only the schema belonging to the ResNet-predicted class.
- `ADDITIONAL`: return `10Additional_documents`; do not extract.
- There is no `UNCERTAIN` state; users can correct the final class or fields in the UI.
- Poor or incomplete OCR does not automatically mean `ADDITIONAL`. Qwen must make
  its best-supported binary decision and report the uncertainty through
  `ocr_quality` for display in the UI.

Only one LLM request is made for each document. The prompt includes one positive
OCR reference and the extraction schema for the ResNet-predicted class only;
references and schemas for unrelated classes are not included. The reference is
guidance, not an exact wording template.

Configuration environment variables:

- `RESNET_CHECKPOINT`: checkpoint path. Defaults to the latest local 11-class model.
- `RESNET_DEVICE`: `auto`, `cpu`, or `cuda`.
- `OLLAMA_BASE_URL`: defaults to `http://127.0.0.1:11434`.
- `COMBINED_LLM_MODEL`: defaults to `qwen3:4b-instruct-2507-q4_K_M`.
- `COMBINED_LLM_TIMEOUT`: defaults to 300 seconds.
- `REFERENCE_OCR_DIR`: directory containing `<ResNet class>/01.json` reference
  OCR files. Defaults to the packaged `assets/reference_ocr` directory.

The ResNet checkpoint is intentionally not stored in Git. Before starting the
website backend, set `RESNET_CHECKPOINT` to the deployed copy of
`resnet18_data2_11class_10Approved_materials_plus_real_02_05.pt`.

---

## 📖 เอกสารและคู่มือการเชื่อมต่อระบบ (Documentation)
- 📘 **คู่มือการเชื่อมต่อ API ละเอียด (สำหรับ Developer / Frontend):** [docs/api_integration_guide.md](docs/api_integration_guide.md)  
  *(ระบุ Input, Output, รายชื่อ Classifier/Models, ขั้นตอน Preprocessing ทั้งหมด, TypeScript Interfaces และตัวอย่างโค้ดเรียกใช้งานใน React Hook, Fetch, Axios, Python, cURL)*
- 🗄️ **แผนภาพโครงสร้างฐานข้อมูล (Database ERD Mermaid):** [docs/database_erd.mmd](docs/database_erd.mmd)
- 📄 **ไฟล์โครงสร้างฐานข้อมูลจริง (PostgreSQL Schema DDL):** [docs/database_schema.sql](docs/database_schema.sql) *(Export สดจาก `pg_dump` ใน Docker Container)*

---

## 🚀 วิธีเปิดใช้งานระบบที่ง่ายที่สุด (1-Click Startup)

ในเครื่อง Windows คุณสามารถเปิดระบบทั้งหมดได้ด้วยการ **ดับเบิ้ลคลิกไฟล์เดียว**:

👉 **ดับเบิ้ลคลิกที่ไฟล์:** `start_server.bat`

> **สิ่งที่ `start_server.bat` ทำให้อัตโนมัติ:**
> 1. ตรวจสอบและสั่งเปิด **PostgreSQL Docker Container** (`expense-reimbursement-postgres`)
> 2. ตรวจสอบและสั่งเปิด **Ollama Local LLM** (Port 11434 ในโหมด CPU)
> 3. เปิด **FastAPI Headless Backend Server** ด้วย Python Environment ใน `.venv`
> 4. เด้งเปิดหน้าเว็บเบราว์เซอร์ไปที่ **Swagger UI (`http://localhost:8000/docs`)** ให้อัตโนมัติ เพื่อทดสอบเรียก API ได้ทันที

---

## 💻 คำสั่งเปิดระบบด้วยตนเองผ่าน Terminal (Manual Commands)

หากต้องการเปิดผ่าน Command Prompt หรือ PowerShell ด้วยตนเอง มีเพียง **3 ขั้นตอน** ดังนี้:

### 1. เปิด PostgreSQL Docker Container
```bash
docker start expense-reimbursement-postgres
```
*(หากยังไม่เคยสร้าง Container มาก่อน ให้รัน: `docker run -d --name expense-reimbursement-postgres -p 5432:5432 -e POSTGRES_PASSWORD=112233 -e POSTGRES_DB=expense_reimbursement_db kxcvbnm/expense-reimbursement-postgres:latest`)*

### 2. เปิด Ollama Service (Local LLM)
```bash
ollama serve
```
*(ทดสอบดูโมเดลที่ติดตั้ง: `ollama list` โดยระบบใช้โมเดล `qwen2.5:3b`)*

### 3. รัน FastAPI Server
```powershell
# รันผ่าน virtual environment ของโปรเจกต์
.\.venv\Scripts\python.exe -m src.app
```

---

## 🌐 ลิงก์และพอร์ตสำคัญของระบบ

| บริการ (Service) | URL / Port | รายละเอียด |
| :--- | :--- | :--- |
| **Interactive API Docs (Swagger)** | [http://localhost:8000/docs](http://localhost:8000/docs) | Swagger UI สำหรับทดสอบ REST API สดผ่านเบราว์เซอร์ |
| **ReDoc API Documentation** | [http://localhost:8000/redoc](http://localhost:8000/redoc) | เอกสาร API สเปก ReDoc ฉบับอ่านง่าย |
| **API Root Status** | [http://localhost:8000](http://localhost:8000) | ข้อมูลสรุปสถานะบริการและรายการ Endpoints (JSON) |
| **Health Check Probe** | [http://localhost:8000/health](http://localhost:8000/health) | Liveness Probe คืนค่า `{"status": "ok"}` |
| **Local LLM (Ollama)** | `http://localhost:11434` | โมเดล Local `qwen2.5:3b` (CPU Mode, Privacy 100% On-Premises) |
| **PostgreSQL Database** | `localhost:5432` | ฐานข้อมูลชุดเรื่อง `expense_reimbursement_db` (User: `postgres`, Pass: `112233`) |

---

---

## 📑 สถาปัตยกรรมและเวิร์กโฟลว์ของระบบ (Workflows)

ระบบกำหนดให้ **PP-ChatOCRv4 เป็นมาตรฐานหลักสำหรับ Production (Standard Workflow)**:

### 🌟 เวิร์กโฟลว์หลัก: PP-ChatOCRv4 (PaddleX + Type-Directed Prompts + Dossier)
* **สกัดข้อมูลแม่แบบ 10 ประเภทอัตโนมัติ:** รองรับ 9 เอกสารเบิกจ่ายราชการ + 1 ใบเสร็จ/ใบกำกับภาษีร้านค้า
* **Multi-page Support:** รองรับเอกสาร PDF หลายหน้าด้วยการต่อภาพแนวตั้งแบบคุมสเกล (≤ 1600 px)
* **Dual Knowledge Payload:** สกัดข้อมูลออกเป็น 2 ส่วนสำหรับ Vector Database
  * `embed_text`: ข้อความสรุปประเด็น Dense Markdown พร้อมเนื้อหาเพื่อทำ Semantic Search
  * `filter_metadata`: ฟิลด์คุณลักษณะสำหรับค้นหาและกรอง (เลขที่, วันที่ ISO, ยอดรวม, สถานะกระทบยอด)
* **Mathematical Reconciliation:** ตรวจกระทบยอดอัตโนมัติ (ยอดก่อนภาษี + VAT = ยอดรวม / สูตรคูณอัตรา x จำนวน)
* **ระบบจัดการชุดเรื่อง (Case Dossier Tracking):** ผูกเอกสารเบิกจ่ายและใบเสร็จเข้ากับ "เอกสารขออนุมัติหลักการ" ใน PostgreSQL อัตโนมัติ พร้อมแดชบอร์ดสรุปงบประมาณคงเหลือ

*(หมายเหตุ: โฟลว์ Custom Pipeline เดิมถูกเก็บไว้สำหรับทดสอบตรวจจับกรอบ Bounding Box และคำนวณแบบแยกขั้นตอนในระดับทดลองเท่านั้น)*

---

## 📋 แม่แบบเอกสารเบิกจ่ายราชการ 9 ประเภท

1. **เอกสารขออนุมัติหลักการ:** เลขที่เอกสาร, วันที่ทำเอกสาร, เรื่อง, ผู้ทำการเบิก (คน/ภาควิชา), รายละเอียดค่าใช้จ่าย, ยอดรวม
2. **เอกสารอนุมัติหลักการ:** รายละเอียดการอนุมัติ, ยอดรวมที่อนุมัติ, ตามหนังสือขออนุมัติหลักการเลขที่เดิม
3. **ขออนุมัติเบิกจ่าย:** เลขที่เอกสาร, วันที่ทำเอกสาร, เรื่อง, รายละเอียดค่าใช้จ่าย, ยอดรวม, ประเภทการเบิก (เงินสดย่อย/ทดรองจ่าย)
4. **แบบเบิกเงินทดรองจ่าย (แบบที่ 1 - ยืมเงิน/เบิก):** เลขที่เอกสาร, ใครเป็นคนเบิก, ยอดรวม, ตามหนังสืออนุมัติเบิกจ่ายเลขที่, รายละเอียดตามประเภทค่าใช้จ่าย, โอนเงินไปที่ใด
5. **แบบเบิกเงินทดรองจ่าย (แบบที่ 2 - รับเงิน/เคลียร์เงิน):** เลขที่เอกสาร, วันที่ส่งเอกสาร, วันที่ขอรับเงิน, ใครเป็นคนเบิก, ยอดรวม, ตามหนังสืออนุมัติหลักการเลขที่, รายละเอียด, โอนเงินไปที่ใด
6. **ใบแทนใบเสร็จ / ใบสำคัญรับเงิน:** วัน/เดือน/ปี, รายละเอียดของรายการการเบิก, ยอดรวม, ผู้จ่ายเงิน
7. **ใบตรวจรับพัสดุ:** เลขที่เอกสาร, ผู้ตรวจรับพัสดุ, วันที่ตรวจรับ, ตามใบสั่งซื้อ/สัญญาเลขที่
8. **ขออนุมัติจัดหาพัสดุ:** เลขที่เอกสาร, วันที่ทำเอกสาร, เรื่อง, เหตุผลที่ต้องจัดหา, รายละเอียดของพัสดุ, วงเงินที่ใช้ทั้งหมด, เวลาที่ต้องใช้พัสดุ
9. **เอกสารประกอบการขออนุมัติจัดหา:** แนบท้ายบันทึกเอกสารเลขอะไร, รายละเอียดพัสดุ/เปรียบเทียบราคา, ยอดรวม

---

## 🛠️ การแก้ปัญหาที่พบบ่อย (Troubleshooting)

### 1. พอร์ต 8000 ชน หรือต้องการปิดเซิร์ฟเวอร์เก่า
หากเซิร์ฟเวอร์เก่าค้างอยู่ สามารถดู Process และปิดได้ด้วย PowerShell:
```powershell
Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
Stop-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess -Force
```

### 2. Docker Container ไม่ยอมสตาร์ต
* ตรวจสอบว่าเปิดโปรแกรม **Docker Desktop** ขึ้นมาแล้วหรือยัง
* ตรวจสอบสถานะ: `docker ps -a`
* สตาร์ตใหม่: `docker start expense-reimbursement-postgres`

### 3. Ollama ไม่ตอบสนอง
* ตรวจสอบว่า Process ทำงานอยู่หรือไม่: `netstat -ano | findstr :11434`
* เปิดใหม่: `ollama serve`
* ตรวจสอบโมเดล: `ollama run qwen2.5:3b "สวัสดี"`
