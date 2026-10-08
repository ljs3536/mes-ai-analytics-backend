FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Seoul \
    HOME=/tmp \
    GIT_PYTHON_REFRESH=quiet

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install lightning==2.4.0

COPY app ./app

RUN useradd --uid 10001 --no-create-home app
USER app

EXPOSE 8003
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8003", "--proxy-headers"]
