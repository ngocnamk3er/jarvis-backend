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

COPY requirements.txt .
# A cache mount rather than --no-cache-dir: this layer is keyed on
# requirements.txt, so adding a single package used to re-download all 122
# of them. Measured 2026-09-26 — adding pypdf and python-docx cost a
# 70-minute rebuild. The cache lives outside the image, so nothing here
# grows it.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# generate_visualization_svg renders its output in headless Chromium before
# returning it (see app/agents/tools/viz_validate.py) — same browser
# `make install-browser` sets up for local dev.
RUN playwright install --with-deps chromium

COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
