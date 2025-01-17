FROM python:3.10-slim

WORKDIR /app

# Install necessary system dependencies
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


COPY requirements.txt /app/

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

EXPOSE 5000

CMD ["python", "main.py"]

#FROM python:alpine3.17
#WORKDIR /app
#COPY . /app
#RUN pip install -r requirements.txt
#ENTRYPOINT ["python", "app.py"]
