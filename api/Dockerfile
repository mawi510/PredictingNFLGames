# Model-serving API for ProMatchPredict.
# Built by CI and deployed to EC2 via docker compose (see infra/).
# Python 3.10: scikit-learn 1.1.2 (the version the model was pickled with) ships
# a cp310 manylinux wheel but NO cp311 wheel, so 3.10 is the newest base that
# pip-installs 1.1.2 cleanly and unpickles the artifact without version warnings.
FROM python:3.10-slim

WORKDIR /app

# Install deps first for layer caching.
COPY api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# App code + the feature manifest. The model.pkl is baked in at build time if
# present, otherwise mounted/pulled from S3 at runtime via MODEL_PATH.
COPY api/ ./api/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
