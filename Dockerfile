# Gunakan base image Python yang sesuai
FROM python:3.12-slim

# Tentukan direktori kerja
WORKDIR /app

# Salin semua file ke dalam container
COPY . /app

# Install dependencies sistem yang diperlukan
RUN apt-get update && apt-get install -y \
    libmariadb-dev-compat \
    libmariadb-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Buat dan aktifkan virtual environment
RUN python -m venv /opt/venv

# Aktifkan virtual environment dan install dependencies Python
RUN . /opt/venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

# Ekspos port aplikasi (opsional)
EXPOSE 8000

# Tentukan perintah default
CMD ["python", "app.py"]