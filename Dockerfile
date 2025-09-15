# Menggunakan base image resmi Python
FROM python:3.10-slim

# Menetapkan direktori kerja di dalam container
WORKDIR /app

# Menyalin file dependensi dan menginstalnya
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Menyalin seluruh kode aplikasi ke dalam direktori kerja
COPY . .

# Menetapkan perintah default yang akan dijalankan saat container dimulai
CMD ["python", "train_model.py"]