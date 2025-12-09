#!/usr/bin/env python3
"""
Split large PDF into <=CHUNK_PAGES parts, upload to GCS, run Document AI batch jobs
(Layout Parser) per chunk, download outputs, and merge Layout into merged_chunks.json.

Assumptions:
 - Service account set via GOOGLE_APPLICATION_CREDENTIALS or Application Default Credentials.
 - .env file contains required variables (see top of file).
"""

import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from pypdf import PdfReader, PdfWriter
from google.cloud import storage
from google.cloud import documentai_v1 as documentai

load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION", "us")
LAYOUT_PROC = os.getenv("LAYOUT_PROCESSOR_ID")
GCS_BUCKET = os.getenv("GCS_BUCKET")
GCS_INPUT_PREFIX = os.getenv("GCS_INPUT_PREFIX", "inputs/tilak")
GCS_OUTPUT_PREFIX = os.getenv("GCS_OUTPUT_PREFIX", "docai-outputs/tilak")
LOCAL_PDF = os.getenv("LOCAL_PDF", "volume.pdf")
CHUNK_PAGES = int(os.getenv("CHUNK_PAGES", "200"))

if not all([PROJECT_ID, LAYOUT_PROC, GCS_BUCKET, LOCAL_PDF]):
    raise SystemExit("Please set PROJECT_ID, LAYOUT_PROCESSOR_ID, GCS_BUCKET and LOCAL_PDF in .env")

# clients
storage_client = storage.Client(project=PROJECT_ID)
docai_client = documentai.DocumentProcessorServiceClient()

def count_pages(pdf_path):
    r = PdfReader(pdf_path)
    return len(r.pages)

def split_pdf(src_pdf_path, chunk_pages=200, out_dir="splits"):
    src = PdfReader(src_pdf_path)
    total = len(src.pages)
    outdir = Path(out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    parts = []
    for start in range(0, total, chunk_pages):
        end = min(start + chunk_pages, total)
        writer = PdfWriter()
        for i in range(start, end):
            writer.add_page(src.pages[i])
        out_path = outdir / f"{Path(src_pdf_path).stem}_part_{start//chunk_pages + 1:03d}.pdf"
        with open(out_path, "wb") as f:
            writer.write(f)
        parts.append(str(out_path))
    return parts

def upload_to_gcs(local_path, bucket_name, dest_prefix):
    bucket = storage_client.bucket(bucket_name)
    dest_name = dest_prefix.rstrip("/") + "/" + Path(local_path).name
    blob = bucket.blob(dest_name)
    blob.upload_from_filename(local_path)
    gs_uri = f"gs://{bucket_name}/{dest_name}"
    print(f"Uploaded {local_path} -> {gs_uri}")
    return gs_uri

def run_batch_and_wait(processor_id, gcs_input_uri, gcs_output_uri, project=PROJECT_ID, location=LOCATION, timeout=3600):
    name = docai_client.processor_path(project, location, processor_id)
    gcs_doc = {"gcs_uri": gcs_input_uri, "mime_type": "application/pdf"}
    request = {
        "name": name,
        "input_documents": {"gcs_documents": {"documents": [gcs_doc]}},
        "document_output_config": {"gcs_output_config": {"gcs_uri": gcs_output_uri}}
    }
    print(f"Starting batch job processor={processor_id} input={gcs_input_uri} output={gcs_output_uri}")
    operation = docai_client.batch_process_documents(request=request)
    print("Operation name:", operation.operation.name)
    try:
        operation.result(timeout=timeout)
    except Exception as e:
        # try to get operation metadata for debug
        try:
            op = docai_client._transport.operations_client.get_operation(operation.operation.name)
            from google.protobuf.json_format import MessageToDict
            print("Operation metadata:", json.dumps(MessageToDict(op.metadata), indent=2)[:5000])
        except Exception as ex:
            print("Could not fetch operation metadata:", repr(ex))
        raise
    print("Batch completed:", operation.operation.name)
    return operation.operation.name

def main():
    total_pages = count_pages(LOCAL_PDF)
    print(f"Local PDF {LOCAL_PDF} has {total_pages} pages.")
    parts = split_pdf(LOCAL_PDF, chunk_pages=CHUNK_PAGES, out_dir="splits")
    print(f"Created {len(parts)} parts.")
    # upload and run per part
    uploaded_gs_uris = []
    for idx, part in enumerate(parts, start=1):
        gs_input_prefix = GCS_INPUT_PREFIX
        gs_uri = upload_to_gcs(part, GCS_BUCKET, gs_input_prefix)
        uploaded_gs_uris.append((part, gs_uri))

    # Run batch jobs per part for Layout
    for idx, (local_part, gs_uri) in enumerate(uploaded_gs_uris, start=1):
        layout_out_prefix = GCS_OUTPUT_PREFIX
        # run Layout Parser
        run_batch_and_wait(LAYOUT_PROC, gs_uri, f"gs://{GCS_BUCKET}/{layout_out_prefix}")


if __name__ == "__main__":
    main()
