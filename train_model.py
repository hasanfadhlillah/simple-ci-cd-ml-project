import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import os
import time

print("===========================================")
print("  Memulai proses pelatihan model ML")
print("===========================================")
print(f"Waktu mulai: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
print(" ")

# Dummy data untuk demonstrasi
print("1. Menyiapkan data dummy...")
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.dot(X, np.array([1, 2])) + 3

print("    Data X:")
print(X)
print("    Data y:")
print(y)
print(" ")

# Membagi data
print("2. Membagi data menjadi training dan testing...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
print(f"    Jumlah data training: {len(X_train)}")
print(f"    Jumlah data testing: {len(X_test)}")
print(" ")

# Melatih model
print("3. Melatih model Linear Regression...")
model = LinearRegression()
model.fit(X_train, y_train)

# Mengecek hasil sederhana
r2_score = model.score(X_test, y_test)
print("    Model selesai dilatih.")
print(f"    Akurasi model (R^2 Score): {r2_score:.2f}")
print(" ")

print("===========================================")
print("  Proses pelatihan model selesai!")
print("===========================================")
print(f"Waktu selesai: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
print(" ")