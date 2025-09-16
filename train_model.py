import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import time
import joblib

print("========================================")
print(" Memulai proses pelatihan model ML")
print("========================================")
print(f"Waktu mulai: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
print(" ")

# 1. Menyiapkan data dummy
print("1. Menyiapkan data dummy...")
X = np.array([
    [1, 1], [1, 2], [2, 2], [2, 3], [3, 3],
    [3, 4], [4, 4], [4, 5], [5, 5], [5, 6]
])
y = np.dot(X, np.array([1, 2])) + 3
print(f"   Jumlah total data: {len(X)} sampel")
print(" ")

# 2. Membagi data
print("2. Membagi data menjadi training dan testing...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"   Jumlah data training: {len(X_train)}")
print(f"   Jumlah data testing: {len(X_test)}")
print(" ")

# 3. Melatih model
print("3. Melatih model Linear Regression...")
model = LinearRegression()
model.fit(X_train, y_train)
r2_score = model.score(X_test, y_test)
print("   Model selesai dilatih.")
print(f"   Akurasi model (R^2 Score): {r2_score:.4f}")
print(" ")

# 4. Menyimpan model
print("4. Menyimpan model ke 'model.joblib'...")
joblib.dump(model, 'model.joblib')
print("   Model berhasil disimpan!")
print(" ")

# 5. Membuat laporan hasil untuk di-deploy
print("5. Membuat file laporan 'report.html'...")
html_content = f"""
<html>
<head>
    <title>Laporan Pelatihan Model</title>
    <style>
        body {{ font-family: sans-serif; margin: 40px; }}
        .container {{ border: 1px solid #ccc; padding: 20px; border-radius: 8px; }}
        h1 {{ color: #333; }}
        p {{ font-size: 1.2em; }}
        strong {{ color: #0056b3; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Laporan Hasil Pelatihan Model ML</h1>
        <p>Model telah berhasil dilatih pada: <strong>{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}</strong></p>
        <p>Akurasi (R² Score) pada data uji: <strong>{r2_score:.4f}</strong></p>
    </div>
</body>
</html>
"""
with open("report.html", "w") as f:
    f.write(html_content)
print("   Laporan berhasil dibuat!")
print(" ")

print("========================================")
print(" Proses pelatihan model selesai!")
print("========================================")