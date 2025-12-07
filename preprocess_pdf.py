#!/usr/bin/env python3
"""
Preprocess PDF pages:
 - convert to 300 DPI images
 - deskew pages
 - despeckle / clean noise
 - optionally recombine cleaned images to a new PDF
 - optional splitting by N pages per chunk (disabled by default for your 117.3MB file)

Usage:
    python preprocess_pdf.py --input path/to/volume.pdf --outdir ./out --recombine_pdf
"""

import os
import sys
import argparse
import math
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm

# -----------------------------
# Utilities: deskew & despeckle
# -----------------------------
def deskew_image_pil(pil_img):
    """Deskew a PIL Image and return deskewed PIL Image."""
    # Convert to grayscale numpy array
    arr = np.array(pil_img.convert("L"))
    # Threshold to binary
    _, bw = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Invert: text -> white on black for finding non-zero points
    bw_inv = 255 - bw

    coords = cv2.findNonZero(bw_inv)  # findNonZero expects single-channel 8-bit
    if coords is None:
        # nothing to deskew
        return pil_img

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    # rectify the angle
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # If angle is tiny, skip rotation
    if abs(angle) < 0.1:
        return pil_img

    (h, w) = arr.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(arr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(rotated)

def despeckle_image_pil(pil_img, median_ksize=3, remove_small_objects=True, min_area=30):
    """Despeckle / denoise image (PIL -> PIL)."""
    arr = np.array(pil_img.convert("L"))
    # Median blur to remove salt-and-pepper
    blurred = cv2.medianBlur(arr, median_ksize)

    # Adaptive threshold to get binary mask (helps to identify small specks)
    th = cv2.adaptiveThreshold(blurred, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)

    # Morphological opening to remove small dots
    kernel = np.ones((1, 1), np.uint8)
    opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)

    # Optionally remove small connected components (speckles)
    if remove_small_objects:
        # find connected components on inverted binary (so text is foreground)
        inv = 255 - opened
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inv, connectivity=8)
        cleaned = np.copy(inv)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area:
                # remove tiny blob
                cleaned[labels == i] = 0
        # invert back to original binary orientation
        cleaned_inv = 255 - cleaned
        # Merge cleaned binary with original grayscale to preserve text tone
        # Use the cleaned_inv mask to sharpen text areas from blurred image
        result = np.where(cleaned_inv == 0, 255, blurred)  # background remain white where removed
        result = result.astype(np.uint8)
    else:
        result = blurred

    return Image.fromarray(result)

# -----------------------------
# Main pipeline
# -----------------------------
def pdf_to_clean_images(pdf_path, outdir, dpi=300, poppler_path=None,
                        deskew=True, despeckle=True, despeckle_params=None):
    """
    Convert PDF to cleaned images and save them into outdir.
    Returns list of saved image paths in order.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Converting PDF to images at {dpi} DPI...")
    convert_kwargs = {"dpi": dpi, "fmt": "png", "output_folder": None}
    if poppler_path:
        convert_kwargs["poppler_path"] = poppler_path

    # convert_from_path returns list of PIL Images (one per page)
    pages = convert_from_path(str(pdf_path), dpi=dpi, fmt="png", poppler_path=poppler_path)
    saved_images = []

    print(f"Processing {len(pages)} pages...")
    for i, pil_page in enumerate(tqdm(pages, desc="pages")):
        page_no = i + 1
        img = pil_page

        if deskew:
            img = deskew_image_pil(img)

        if despeckle:
            params = despeckle_params or {}
            img = despeckle_image_pil(img,
                                     median_ksize=params.get("median_ksize", 3),
                                     remove_small_objects=params.get("remove_small_objects", True),
                                     min_area=params.get("min_area", 30))

        out_path = outdir / f"page_{page_no:04d}.png"
        img.save(out_path, format="PNG")
        saved_images.append(str(out_path))

    return saved_images

def images_to_pdf(image_paths, out_pdf_path):
    """Combine image PNGs into multi-page PDF using Pillow."""
    if not image_paths:
        raise ValueError("No images to write to PDF")
    pil_imgs = [Image.open(p).convert("RGB") for p in image_paths]
    first, rest = pil_imgs[0], pil_imgs[1:]
    first.save(out_pdf_path, save_all=True, append_images=rest, quality=95)
    print(f"Saved cleaned PDF to: {out_pdf_path}")
    # close images
    for im in pil_imgs:
        im.close()

def split_pdf_pages(pdf_path, chunk_size, outdir):
    """
    Split PDF into multiple PDFs each with chunk_size pages.
    Uses pdf2image to render and re-create. This is a simple approach,
    better approaches use PyPDF2 to split without image-roundtrips.
    """
    # For splitting without rasterizing, use PyPDF2 or pypdf (recommended).
    try:
        from PyPDF2 import PdfReader, PdfWriter
    except ImportError:
        raise RuntimeError("PyPDF2 is required for PDF splitting. pip install PyPDF2")

    reader = PdfReader(str(pdf_path))
    total = len(reader.pages)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    parts = []
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        writer = PdfWriter()
        for p in range(start, end):
            writer.add_page(reader.pages[p])
        out_pdf = outdir / f"{Path(pdf_path).stem}_part_{start//chunk_size + 1:03d}.pdf"
        with open(out_pdf, "wb") as f:
            writer.write(f)
        parts.append(str(out_pdf))
    return parts

# -----------------------------
# CLI entrypoint
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Preprocess PDF: DPI, deskew, despeckle, optional split")
    p.add_argument("--input", "-i", required=True, help="Path to input PDF")
    p.add_argument("--outdir", "-o", required=True, help="Directory to write processed images / PDF")
    p.add_argument("--dpi", type=int, default=300, help="DPI to render PDF (default 300)")
    p.add_argument("--poppler-path", default=None, help="Poppler bin path (Windows).")
    p.add_argument("--no-deskew", action="store_true", help="Disable deskew")
    p.add_argument("--no-despeckle", action="store_true", help="Disable despeckle")
    p.add_argument("--recombine-pdf", dest="recombine_pdf", action="store_true",
                   help="Recombine cleaned pages into output PDF")
    p.add_argument("--split-if-larger-than-mb", type=int, default=200,
                   help="If input PDF size (MB) > this, split into chunks (default 200 MB).")
    p.add_argument("--chunk-pages", type=int, default=100,
                   help="If splitting, split into chunks of this many pages (default 100).")
    return p.parse_args()

def main():
    args = parse_args()
    pdf_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not pdf_path.exists():
        print("Input PDF not found:", pdf_path)
        sys.exit(1)

    filesize_mb = pdf_path.stat().st_size / (1024 * 1024)
    print(f"Input file: {pdf_path} ({filesize_mb:.1f} MB)")

    # Decision based on your filesize: user said 117.3 MB
    if filesize_mb > args.split_if_larger_than_mb:
        print(f"PDF over {args.split_if_larger_than_mb} MB -> splitting into chunks of {args.chunk_pages} pages")
        parts = split_pdf_pages(pdf_path, args.chunk_pages, outdir / "splits")
        print("Created split PDFs:", parts)
        # Optionally: process each part separately (not implemented here)
    else:
        print(f"PDF size ({filesize_mb:.1f} MB) is under threshold ({args.split_if_larger_than_mb} MB). Skipping splitting.")

    print("Starting page-level preprocessing...")
    saved_images = pdf_to_clean_images(pdf_path, outdir / "images", dpi=args.dpi,
                                       poppler_path=args.poppler_path,
                                       deskew=not args.no_deskew,
                                       despeckle=not args.no_despeckle,
                                       despeckle_params={"median_ksize": 3, "remove_small_objects": True, "min_area": 30})

    if args.recombine_pdf:
        out_pdf_path = outdir / f"{pdf_path.stem}_cleaned.pdf"
        images_to_pdf(saved_images, out_pdf_path)

    print("Done. Cleaned images saved to:", outdir / "images")

if __name__ == "__main__":
    main()
