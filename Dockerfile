# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# This host's IPv6 path is broken (confirmed: `curl -6` fails outright,
# `curl -4` succeeds instantly) — glibc's resolver still races an IPv6
# attempt first by default and eats the whole timeout budget before
# falling back, intermittently killing pip/apt downloads mid-build.
# Standard fix: make getaddrinfo prefer IPv4-mapped addresses.
RUN echo "precedence ::ffff:0:0/96  100" >> /etc/gai.conf

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# LibreOffice, run headless by app/agents/office.py: converts Word/Excel/
# PowerPoint files (office_convert), recalculates workbooks (office_recalculate)
# and draws their pages (view_image). The agent's sandbox has no library for
# these formats, so the server does it. Its own layer, ahead of the pip layers,
# so a change to requirements.txt does not reinstall it. The fonts are what the
# rendered pages use; without them text falls back to a default that changes
# the layout.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libreoffice-writer libreoffice-calc libreoffice-impress \
    fonts-dejavu-core fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

# tesseract, run by app/agents/ocr.py (ocr_image / draw_boxes): where the text sits in an
# image, so boxes drawn on it land on the text instead of where a model guessed. Its own
# layer, after LibreOffice, so adding or changing it does not reinstall that.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-eng tesseract-ocr-vie \
    && rm -rf /var/lib/apt/lists/*

# More languages for ocr_image's `lang` parameter. Another layer again, so the ones above stay cached.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr-jpn tesseract-ocr-chi-sim tesseract-ocr-chi-tra tesseract-ocr-kor tesseract-ocr-rus tesseract-ocr-tha tesseract-ocr-ara tesseract-ocr-fra tesseract-ocr-deu tesseract-ocr-spa \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# A cache mount rather than --no-cache-dir: this layer is keyed on
# requirements.txt, so adding a single package used to re-download all 122
# of them. Measured 2026-09-26 — adding pypdf and python-docx cost a
# 70-minute rebuild. The cache lives outside the image, so nothing here
# grows it.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# playwright's bundled node driver links against libatomic, which
# python:3.11-slim doesn't ship; without it `playwright install` below dies
# with exit 127 ("libatomic.so.1: cannot open shared object file") — hit on
# 2026-11-11 once requirements.txt's unpinned playwright resolved to 1.64.1.
# A step of its own, after the pip layer, so adding it doesn't invalidate
# the 15-minute pip install above.
RUN apt-get update && apt-get install -y --no-install-recommends libatomic1 \
    && rm -rf /var/lib/apt/lists/*

# generate_visualization_svg renders its output in headless Chromium before
# returning it (see app/agents/tools/viz_validate.py) — same browser
# `make install-browser` sets up for local dev.
RUN playwright install --with-deps chromium

# PaddleOCR, the engine behind ocr_image (app/agents/ocr_paddle.py): much better than tesseract on
# small, noisy or tilted text. Versions pinned to what jarvis-ocr-worker runs. Large (paddle alone
# unpacks to ~700MB), so a layer of its own with a pip cache: nothing above it is reinstalled when
# it changes. paddlex[ocr-core] is what its OCR pipelines check for (without it the models cannot be
# created, and that check looks for the package named opencv-contrib-python in particular). That
# opencv and the headless one from requirements.txt both provide `cv2`, so the headless one goes and
# the contrib one is put back over it, leaving a single cv2 (it needs the three libraries below).
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "paddlepaddle==3.3.1" "paddleocr==3.7.0" "paddlex[ocr-core]==3.7.2" \
    && pip uninstall -y opencv-python-headless \
    && pip install --force-reinstall --no-deps "opencv-contrib-python==4.10.0.84"

# Bake the recognition models in, as the OCR worker does: otherwise the first ocr_image after a
# pod start waits to download them, and the sandboxed server may not be allowed to reach the model
# host. A failure here is tolerated (the runtime download remains the fallback).
RUN python3 -c "\
from paddleocr import PaddleOCR; \
[PaddleOCR(lang=l, use_textline_orientation=False, use_doc_orientation_classify=False, use_doc_unwarping=False, enable_mkldnn=False) \
 for l in ('vi', 'en', 'japan', 'ch', 'chinese_cht', 'korean', 'ru', 'th', 'ar')]" || echo 'WARNING: OCR models were not baked in; they will be downloaded on first use'

COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
