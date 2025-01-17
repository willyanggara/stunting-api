# Gunakan image Python
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Salin file requirements
COPY requirements.txt /app/

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file aplikasi, termasuk .env
COPY . /app/

# Buat folder /static jika tidak ada
RUN mkdir -p /app/static

# Expose port
EXPOSE 5000

# Jalankan aplikasi
CMD ["python", "main.py"]
