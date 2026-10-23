FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir transformers peft fastapi uvicorn
COPY src ./src
COPY artifacts/lora ./artifacts/lora
COPY artifacts/model_int8.pt ./artifacts/model_int8.pt
EXPOSE 8000
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
