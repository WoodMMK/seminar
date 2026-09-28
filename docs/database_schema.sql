-- =====================================================================
-- Thai Financial Document OCR & Expense Reimbursement System
-- Production PostgreSQL Database DDL Schema
-- Database: expense_reimbursement_db
-- Container: expense-reimbursement-postgres (PostgreSQL 16)
-- Paired with: docs/database_erd.mmd
-- =====================================================================

-- 1. Enable Cryptographic Functions Extension for UUID generation
CREATE EXTENSION IF NOT EXISTS pgcrypto WITH SCHEMA public;
COMMENT ON EXTENSION pgcrypto IS 'cryptographic functions for gen_random_uuid()';

SET statement_timeout = 0;
SET lock_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SET check_function_bodies = false;
SET client_min_messages = warning;

-- =====================================================================
-- Table 1: users (ผู้ใช้งานและเจ้าหน้าที่ในระบบ)
-- =====================================================================
CREATE TABLE public.users (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    username character varying(50) NOT NULL,
    email character varying(100) NOT NULL,
    role character varying(20) NOT NULL,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    CONSTRAINT users_pkey PRIMARY KEY (id),
    CONSTRAINT uq_users_username UNIQUE (username),
    CONSTRAINT uq_users_email UNIQUE (email)
);

COMMENT ON TABLE public.users IS 'บัญชีผู้ใช้งาน เจ้าหน้าที่การเงิน และผู้ตรวจสอบระบบ';
COMMENT ON COLUMN public.users.id IS 'Primary Key (UUID)';
COMMENT ON COLUMN public.users.role IS 'บทบาทผู้ใช้ เช่น officer, auditor, admin';

-- =====================================================================
-- Table 2: disbursement_records (ชุดเรื่องการเบิกจ่าย / Case Dossier)
-- =====================================================================
CREATE TABLE public.disbursement_records (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    user_id uuid NOT NULL,
    principle_doc_no character varying(100),
    title character varying(255),
    record_type character varying(50) NOT NULL,
    receiver_name character varying(50) NOT NULL,
    approved_amount numeric(12,2) NOT NULL,
    actual_expense numeric(12,2) NOT NULL,
    status character varying(50) NOT NULL,
    request_date timestamp with time zone NOT NULL,
    due_date timestamp with time zone NOT NULL,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    CONSTRAINT disbursement_records_pkey PRIMARY KEY (id),
    CONSTRAINT fk_disbursement_records_user FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE RESTRICT
);

COMMENT ON TABLE public.disbursement_records IS 'ชุดเรื่องหลักการเบิกจ่าย (Case Dossier) ที่ผูกเอกสารหลักการ ใบเสร็จ และรายการเบิกเข้าด้วยกัน';
COMMENT ON COLUMN public.disbursement_records.id IS 'Primary Key (Case ID)';
COMMENT ON COLUMN public.disbursement_records.principle_doc_no IS 'เลขที่หนังสือขออนุมัติหลักการต้นเรื่อง (เช่น อว 78.101/20271)';
COMMENT ON COLUMN public.disbursement_records.approved_amount IS 'วงเงินงบประมาณที่ได้รับอนุมัติตามหลักการ (บาท)';
COMMENT ON COLUMN public.disbursement_records.actual_expense IS 'ยอดเงินเบิกจ่ายจริงสะสมในชุดเรื่อง (บาท)';

-- =====================================================================
-- Table 3: documents (เอกสารที่แนบในชุดเรื่อง พร้อมผลลัพธ์ OCR & AI JSONB)
-- =====================================================================
CREATE TABLE public.documents (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    record_id uuid NOT NULL,
    filename character varying(150) NOT NULL,
    s3_path character varying(300) NOT NULL,
    document_type character varying(50) NOT NULL,
    classified_step character varying(50),
    extracted_data jsonb,
    uploaded_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    CONSTRAINT documents_pkey PRIMARY KEY (id),
    CONSTRAINT uq_documents_s3_path UNIQUE (s3_path),
    CONSTRAINT fk_documents_record FOREIGN KEY (record_id) REFERENCES public.disbursement_records(id) ON DELETE CASCADE
);

COMMENT ON TABLE public.documents IS 'เอกสารต้นฉบับ ไฟล์แนบ และข้อมูลทางการเงินที่สกัดได้จาก PP-ChatOCRv4 ในรูป JSONB';
COMMENT ON COLUMN public.documents.id IS 'Primary Key (Document ID)';
COMMENT ON COLUMN public.documents.record_id IS 'Foreign Key อ้างอิง Case Dossier (disbursement_records.id)';
COMMENT ON COLUMN public.documents.extracted_data IS 'ผลลัพธ์ที่สกัดได้แบบสมบูรณ์: chat_answers, embed_text, filter_metadata, math_reconciliation';

-- =====================================================================
-- Table 4: expense_items (รายการค่าใช้จ่ายย่อยในแต่ละเอกสาร)
-- =====================================================================
CREATE TABLE public.expense_items (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    record_id uuid NOT NULL,
    document_id uuid,
    item_name character varying(50) NOT NULL,
    quantity integer,
    unit_price numeric(12,2),
    total_amount numeric(12,2) NOT NULL,
    expense_type character varying(50) NOT NULL,
    CONSTRAINT expense_items_pkey PRIMARY KEY (id),
    CONSTRAINT fk_expense_items_record FOREIGN KEY (record_id) REFERENCES public.disbursement_records(id) ON DELETE CASCADE,
    CONSTRAINT fk_expense_items_document FOREIGN KEY (document_id) REFERENCES public.documents(id) ON DELETE SET NULL
);

COMMENT ON TABLE public.expense_items IS 'รายการแจกแจงค่าใช้จ่ายย่อย เช่น ค่าอาหารว่าง, ค่าวัสดุ, ค่าตอบแทนวิทยากร';
COMMENT ON COLUMN public.expense_items.id IS 'Primary Key (Item ID)';
COMMENT ON COLUMN public.expense_items.record_id IS 'Foreign Key อ้างอิง Case Dossier';
COMMENT ON COLUMN public.expense_items.document_id IS 'Foreign Key อ้างอิงเอกสารต้นทางของรายการ';

-- =====================================================================
-- Table 5: work_logs (บันทึกประวัติการอนุมัติและการตรวจสอบสถานะ)
-- =====================================================================
CREATE TABLE public.work_logs (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    record_id uuid NOT NULL,
    changed_by uuid NOT NULL,
    previous_status character varying(50) NOT NULL,
    new_status character varying(50) NOT NULL,
    notes text,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    CONSTRAINT work_logs_pkey PRIMARY KEY (id),
    CONSTRAINT fk_work_logs_record FOREIGN KEY (record_id) REFERENCES public.disbursement_records(id) ON DELETE CASCADE,
    CONSTRAINT fk_work_logs_changed_by FOREIGN KEY (changed_by) REFERENCES public.users(id) ON DELETE RESTRICT
);

COMMENT ON TABLE public.work_logs IS 'บันทึก Audit Trail การเปลี่ยนสถานะและข้อเสนอแนะในการตรวจสอบเอกสาร';

-- =====================================================================
-- Indexes for High Performance Queries & Search
-- =====================================================================
CREATE INDEX idx_disbursement_records_principle ON public.disbursement_records USING btree (principle_doc_no);
CREATE INDEX idx_disbursement_records_status ON public.disbursement_records USING btree (status);
CREATE INDEX idx_disbursement_records_user_id ON public.disbursement_records USING btree (user_id);

CREATE INDEX idx_documents_record_id ON public.documents USING btree (record_id);
CREATE INDEX idx_documents_extracted_data ON public.documents USING gin (extracted_data);

CREATE INDEX idx_expense_items_record_id ON public.expense_items USING btree (record_id);
CREATE INDEX idx_expense_items_document_id ON public.expense_items USING btree (document_id);

CREATE INDEX idx_work_logs_record_id ON public.work_logs USING btree (record_id);
CREATE INDEX idx_work_logs_changed_by ON public.work_logs USING btree (changed_by);
