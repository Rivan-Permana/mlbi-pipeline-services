# ---------------------------------------------------------
# Python 3.11 slim — Cloud Run friendly
# Jalankan FastAPI via gunicorn + uvicorn worker
# ---------------------------------------------------------
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

# (opsional) user non-root
RUN addgroup --system app && adduser --system --ingroup app app

# System deps ringan; tambahkan lain jika perlu (libgomp1, libstdc++6)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
  && rm -rf /var/lib/apt/lists/*

# Copy requirements lebih dulu untuk caching layer
COPY requirements.txt .

# Install python deps dari requirements + gunicorn (server)
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn

# Copy source code
COPY main.py .

# Temp dir (Cloud Run writable dirs: /tmp)
RUN mkdir -p /tmp && chown -R app:app /app /tmp

USER app

# EXPOSE tidak wajib di Cloud Run, tapi aman
EXPOSE 8080

# Jalankan app: objek FastAPI ada di main.py → app
# Cloud Run akan menyuplai $PORT; bind ke 0.0.0.0:$PORT
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "2", "-b", "0.0.0.0:${PORT}", "main:app"]
