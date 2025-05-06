FROM python:3.12-alpine

WORKDIR /app

# Cài đặt thư viện
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Tạo thư mục log
RUN mkdir -p logs

CMD ["python", "main.py"]
