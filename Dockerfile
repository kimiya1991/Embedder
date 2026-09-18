FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY lab.py ui.py sample.txt /app/
COPY .streamlit /app/.streamlit

ENV PYTHONUNBUFFERED=1

EXPOSE 8501

CMD ["streamlit", "run", "ui.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
