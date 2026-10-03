# Thai Financial Document Extraction REST API Backend

ระบบสกัดข้อมูลและตรวจสอบความถูกต้องของเอกสารเบิกจ่ายราชการและเอกสารการเงินภาษาไทยอัตโนมัติ (**Headless FastAPI Backend Service** ขับเคลื่อนด้วย PP-ChatOCRv4, Thai PP-OCRv5 และ Local LLM)

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
> 1. ตรวจจับการมีอยู่ของ **Hardware GPU (NVIDIA CUDA / ROCm / Metal)** และเปิดใช้งาน `OLLAMA_NUM_GPU=-1` พร้อม `FlashAttention` ให้อัตโนมัติ (หากไม่มี GPU จะสลับรันโหมด CPU ได้อย่างราบรื่น)
> 2. ตรวจสอบและสั่งเปิด **PostgreSQL Docker Container** (`expense-reimbursement-postgres`)
> 3. ตรวจสอบและสั่งเปิด **Ollama Local LLM** (Port 11434 พร้อมตั้งค่าโหลดขึ้น VRAM ของ GPU เต็มประสิทธิภาพ)
> 4. เปิด **FastAPI Headless Backend Server** ด้วย Python Environment ใน `.venv`
> 5. เด้งเปิดหน้าเว็บเบราว์เซอร์ไปที่ **Swagger UI (`http://localhost:8000/docs`)** ให้อัตโนมัติ เพื่อทดสอบเรียก API ได้ทันที

---

## ⚡ การประมวลผลผ่าน GPU (Hardware & GPU Acceleration)

ระบบออกแบบมาให้ **Plug-and-Play สำหรับทุกคนในทีม**:
* **เครื่องที่มี GPU (NVIDIA GeForce/RTX, AMD, Apple Silicon):**
  เมื่อรัน `start_server.bat` หรือยิง API ระบบจะส่งพารามิเตอร์ `num_gpu: -1` ไปยัง Ollama ซึ่งเป็นการสั่งให้ **Offload เลเยอร์ทั้งหมดของโมเดล (100% Layers) ขึ้น VRAM ของ GPU** ทันที ทำให้ประมวลผลสกัดข้อมูลได้เร็วในระดับเสี้ยววินาที (< 1-2 วินาที)
* **เครื่องที่ไม่มี GPU (CPU Only):**
  Ollama และ ONNX Perception Engine จะตรวจจับและ Fallback กลับมาประมวลผลบน CPU อัตโนมัติ โดยไม่ต้องแก้ไขโค้ดหรือคอนฟิกใดๆ

#### วิธีตรวจสอบว่าโมเดลกำลังรันบน GPU หรือไม่:
เปิด Terminal และพิมพ์:
```bash
ollama ps
```
ผลลัพธ์จะแสดงคอลัมน์ `PROCESSOR` เช่น `100% GPU` (หรือ `100% CPU` หากเครื่องไม่มี GPU)

#### การปรับแต่งตัวแปรสภาพแวดล้อม (Environment Variables):
| ตัวแปร (Variable) | ค่าเริ่มต้น (Default) | คำอธิบาย |
| :--- | :---: | :--- |
| `OLLAMA_NUM_GPU` | `-1` | จำนวนเลเยอร์ที่ส่งไป GPU (`-1` = โหลดทุกเลเยอร์ขึ้น GPU ทั้งหมด, `0` = บังคับใช้เฉพาะ CPU) |
| `OCR_DEVICE` | `cpu` / `gpu` | เลือกรันโมเดล OCR ด้วย CPU หรือ GPU |
| `OCR_USE_ONNX` | `true` | เปิดใช้งาน ONNX Runtime สำหรับ Text Detection & Recognition ความเร็วสูง |

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
| **Local LLM (Ollama)** | `http://localhost:11434` | โมเดล Local `qwen2.5:3b` (Adaptive GPU/CPU Mode, Privacy 100% On-Premises) |
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
