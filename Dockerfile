FROM python:3.11-slim

WORKDIR /app
COPY serve.py .

RUN pip install --no-cache-dir fastapi uvicorn transformers[torch] pillow
# Pre-download model during build
RUN python -c "from transformers import pipeline; pipeline('image-classification', model='prithivMLmods/Deep-Fake-Detector-v2-Model')"

EXPOSE 8080
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8080"]
