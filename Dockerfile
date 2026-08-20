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
RUN pip install --no-cache-dir -r requirements.txt

# generate_visualization_svg renders its output in headless Chromium before
# returning it (see app/agents/tools/viz_validate.py) — same browser
# `make install-browser` sets up for local dev.
RUN playwright install --with-deps chromium

COPY alembic.ini .
COPY migrations ./migrations
COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
