FROM pytorch/pytorch:2.9.1-cuda13.0-cudnn9-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

EXPOSE 8765

ENTRYPOINT ["python", "-u", "-m", "src.main"]
