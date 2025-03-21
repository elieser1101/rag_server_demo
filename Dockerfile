# Use official Python image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy required files
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY main.py .

EXPOSE 8000

ENV APP_WORKERS=${APP_WORKERS:-1}

CMD uvicorn main:app --host 0.0.0.0 --port 8000 --workers ${APP_WORKERS}