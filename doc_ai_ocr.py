#!/usr/bin/env python3
"""
Batch-run Document AI processors (OCR + Layout Parser) on a PDF in GCS,
download Document JSON outputs, and merge layout blocks -> OCR text.

Assumes Document AI BatchProcess was used and outputs JSON files to GCS_OUTPUT_URI.
"""

import os
import json
import time
from pathlib import Path
from google.cloud import documentai_v1 as documentai
from google.cloud import storage
from dotenv import load_dotenv

load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION", "us")
OCR_PROCESSOR_ID = os.getenv("OCR_PROCESSOR_ID")
LAYOUT_PROCESSOR_ID = os.getenv("LAYOUT_PROCESSOR_ID")
GCS_URI = os.getenv("GCS_URI")  # e.g. gs://your-bucket/volume.pdf
GCS_OUTPUT_URI = os.getenv("GCS_OUTPUT_URI")  # e.g. gs://your-bucket/docai-outputs/volume1/

if not all([PROJECT_ID, OCR_PROCESSOR_ID, LAYOUT_PROCESSOR_ID, GCS_URI, GCS_OUTPUT_URI]):
    raise SystemExit("Please set PROJECT_ID, OCR_PROCESSOR_ID, LAYOUT_PROCESSOR_ID, GCS_URI, GCS_OUTPUT_URI in .env")

# Clients
docai_client = documentai.DocumentProcessorServiceClient()
storage_client = storage.Client(project=PROJECT_ID)

def run_batch(processor_id, gcs_input_uri, gcs_output_uri, project=PROJECT_ID, location=LOCATION):
    """
    Run batch_process_documents for the given processor and wait for completion.
    Returns the operation response (the Operation).
    """
    name = docai_client.processor_path(project, location, processor_id)
    gcs_doc = {"gcs_uri": gcs_input_uri, "mime_type": "application/pdf"}

    # Build request dict (Document AI client accepts dict)
    request = {
        "name": name,
        "input_documents": {"gcs_documents": {"documents": [gcs_doc]}},
        "document_output_config": {"gcs_output_config": {"gcs_uri": gcs_output_uri}}
    }

    print(f"Starting batch job for processor {processor_id} -> output: {gcs_output_uri}")
    operation = docai_client.batch_process_documents(request=request)
    operation.result(timeout=3600)  # waits (adjust timeout if you expect >1h)
    print("Batch job finished.")
    return operation

def list_gcs_outputs(gcs_output_uri):
    """List JSON output files created by Document AI in the given GCS prefix."""
    # gcs_output_uri format: gs://bucket/path/to/prefix/
    assert gcs_output_uri.startswith("gs://")
    parts = gcs_output_uri[5:].split("/", 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    bucket = storage_client.bucket(bucket_name)
    blobs = list(storage_client.list_blobs(bucket_name, prefix=prefix))
    # Document AI writes files like output-1-to-1.json inside an operation folder; include only .json
    json_blobs = [b for b in blobs if b.name.endswith(".json")]
    print(f"Found {len(json_blobs)} JSON output file(s) in {gcs_output_uri}")
    return bucket_name, [b.name for b in json_blobs]

def download_and_load_json(bucket_name, blob_name, dst_dir="docai_json"):
    Path(dst_dir).mkdir(parents=True, exist_ok=True)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    dst_path = Path(dst_dir) / Path(blob_name).name
    blob.download_to_filename(str(dst_path))
    with open(dst_path, "r", encoding="utf-8") as f:
        doc_json = json.load(f)
    return doc_json, str(dst_path)

# -------------------------
# Helpers to extract text from textAnchor (JSON)
# -------------------------
def get_text_from_text_anchor_json(document_json, text_anchor):
    """
    document_json: the full Document JSON as returned by Document AI
    text_anchor: dict that contains 'textSegments' list with 'startIndex'/'endIndex'
    """
    if not text_anchor:
        return ""
    text = document_json.get("text", "")
    out = []
    for seg in text_anchor.get("textSegments", []):
        # startIndex might be string in JSON; convert
        start = int(seg.get("startIndex", 0))
        end = int(seg.get("endIndex", len(text)))
        out.append(text[start:end])
    return "".join(out)

# -------------------------
# Merge logic (works with Document AI JSON structure)
# -------------------------
def merge_layout_and_ocr_from_json(ocr_doc_json, layout_doc_json):
    """
    Given OCR document JSON and Layout Parser document JSON, return merged chunks.
    Returns list of chunks: {'type':..., 'page':n, 'text':..., 'bbox':...}
    """
    chunks = []
    # Layout parser outputs blocks under pages[].blocks (name may vary slightly depending on processor)
    pages = layout_doc_json.get("pages", [])
    for page_idx, page in enumerate(pages):
        # Blocks (if present)
        for block in page.get("blocks", []):
            layout = block.get("layout", {})
            text_anchor = layout.get("textAnchor")
            text = get_text_from_text_anchor_json(ocr_doc_json, text_anchor)
            btype = block.get("type", "PARAGRAPH")
            bbox = layout.get("boundingPoly", {})
            chunks.append({
                "type": btype,
                "page": page_idx + 1,
                "text": text.strip(),
                "bbox": bbox
            })

        # Tables
        for table in page.get("tables", []):
            # Document AI JSON tables have headerRows and bodyRows structures
            rows = []
            # headerRows might be under table.get("headerRows", []), bodyRows under table.get("bodyRows", [])
            header_rows = table.get("headerRows", [])
            body_rows = table.get("bodyRows", [])
            for r in header_rows + body_rows:
                row_cells = []
                for c in r.get("cells", []):
                    # cell.layout.textAnchor
                    cell_text = get_text_from_text_anchor_json(ocr_doc_json, c.get("layout", {}).get("textAnchor"))
                    row_cells.append(cell_text.strip())
                rows.append(row_cells)
            chunks.append({
                "type": "TABLE",
                "page": page_idx + 1,
                "table": rows
            })
    return chunks

# -------------------------
# Main flow
# -------------------------
def main():
    # 1) Run OCR batch job
    ocr_output_prefix = os.path.join(GCS_OUTPUT_URI.rstrip("/"), "ocr/")
    run_batch(OCR_PROCESSOR_ID, GCS_URI, ocr_output_prefix)

    # 2) Run Layout Parser batch job (separate output prefix)
    layout_output_prefix = os.path.join(GCS_OUTPUT_URI.rstrip("/"), "layout/")
    run_batch(LAYOUT_PROCESSOR_ID, GCS_URI, layout_output_prefix)

    # 3) List & download outputs for OCR
    time.sleep(2)
    bucket_name_ocr, ocr_json_blobs = list_gcs_outputs(ocr_output_prefix)
    if not ocr_json_blobs:
        raise SystemExit("No OCR output JSON files found. Check GCS_OUTPUT_URI and permissions.")
    # download first JSON (Document AI may produce many; here we assume 1)
    ocr_doc_json, ocr_json_path = download_and_load_json(bucket_name_ocr, ocr_json_blobs[0], dst_dir="outputs/ocr")
    print("Downloaded OCR JSON to:", ocr_json_path)

    # 4) List & download layout outputs
    bucket_name_layout, layout_json_blobs = list_gcs_outputs(layout_output_prefix)
    if not layout_json_blobs:
        raise SystemExit("No Layout output JSON files found. Check GCS_OUTPUT_URI and permissions.")
    layout_doc_json, layout_json_path = download_and_load_json(bucket_name_layout, layout_json_blobs[0], dst_dir="outputs/layout")
    print("Downloaded Layout JSON to:", layout_json_path)

    # 5) Merge into chunks
    print("Merging layout and OCR into chunks...")
    chunks = merge_layout_and_ocr_from_json(ocr_doc_json.get("document", ocr_doc_json), layout_doc_json.get("document", layout_doc_json))
    # The JSON file might either be the raw Document or contain top-level {"document": { ... }} depending on how you saved outputs.
    # The helper above tries to be resilient to both.

    print(f"Found {len(chunks)} chunks. Printing first 10 (type, page, first 200 chars):")
    for c in chunks[:10]:
        t = c.get("text", "")
        typ = c.get("type")
        page = c.get("page")
        preview = (t[:200] + "...") if len(t) > 200 else t
        print(f" - [{typ}] page={page} -> {preview}")

    # Optionally: save merged chunks to a JSON file
    out_path = Path("merged_chunks.json")
    out_path.write_text(json.dumps(chunks, indent=2, ensure_ascii=False))
    print("Merged chunks saved to:", out_path)

if __name__ == "__main__":
    main()
