# Model-serving API for ProMatchPredict.
# Built by CI and deployed to EC2 via docker compose (see infra/).
# Python 3.10: matches the weekly-data job's pinned stack (numpy 1.23.5 has no
# constraint newer than cp310 here); xgboost 2.1.3 ships a cp310 wheel.
FROM python:3.10-slim

WORKDIR /app

# Install deps first for layer caching, then strip weight we don't ship:
# bundled test suites in numpy/scipy/sklearn, bytecode, and pip/setuptools.
COPY api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
    && find /usr/local/lib/python3.10/site-packages \
        -type d \( -name tests -o -name test \) -prune -exec rm -rf {} + \
    && find /usr/local/lib/python3.10/site-packages \
        -name "*.pyc" -delete \
    && pip uninstall -y pip setuptools wheel 2>/dev/null || true

# App code + feature manifest + the committed margin-v2 model bundle (small
# enough to live in the repo — no S3 pull at boot).
COPY api/ ./api/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
