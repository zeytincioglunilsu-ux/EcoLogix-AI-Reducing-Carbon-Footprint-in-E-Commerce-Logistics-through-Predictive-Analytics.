# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:05:35 2026

@author: Elif Elvin Acar
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('ecommerce_logistics_carbon_emissions_v1.csv')

# 1. Temel Bilgiler ve İstatistikler
print("--- Veri Seti Özeti ---")
print(df.info())
print("\n--- Betimsel İstatistikler ---")
print(df.describe())
print("\n--- Eksik Değer Kontrolü ---")
print(df.isnull().sum())

# Renk paleti tanımlayalım
colors = ['#4c72b0', '#55a868', '#c44e52', '#8172b2', '#ccb974', '#64b5cd']

# 2. Karbon Salınımı Dağılımı (Histogram)
plt.figure(figsize=(10, 6))
plt.hist(df['Carbon_Emission_kgCO2e'], bins=50, color='skyblue', edgecolor='black')
plt.title('Karbon Salınımı Dağılımı (kgCO2e)')
plt.xlabel('Karbon Salınımı')
plt.ylabel('Frekans')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# 3. Araç Tiplerinin Kullanım Sıklığı (Bar Chart)
plt.figure(figsize=(12, 6))
vehicle_counts = df['Vehicle_Type'].value_counts()
plt.bar(vehicle_counts.index, vehicle_counts.values, color=colors[:len(vehicle_counts)])
plt.title('Araç Tiplerinin Kullanım Sıklığı')
plt.xlabel('Araç Tipi')
plt.ylabel('Adet')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 4. Mesafe ve Karbon Salınımı İlişkisi (Scatter Plot)
plt.figure(figsize=(10, 6))
plt.scatter(df['Distance_KM'], df['Carbon_Emission_kgCO2e'], alpha=0.5, color='orange')
plt.title('Mesafe (KM) vs Karbon Salınımı')
plt.xlabel('Mesafe (KM)')
plt.ylabel('Karbon Salınımı (kgCO2e)')
plt.show()

# 5. Araç Tipine Göre Ortalama Karbon Salınımı (Bar Chart)
plt.figure(figsize=(12, 6))
avg_carbon_vehicle = df.groupby('Vehicle_Type')['Carbon_Emission_kgCO2e'].mean().sort_values(ascending=False)
plt.bar(avg_carbon_vehicle.index, avg_carbon_vehicle.values, color='salmon')
plt.title('Araç Tipine Göre Ortalama Karbon Salınımı')
plt.xlabel('Araç Tipi')
plt.ylabel('Ortalama kgCO2e')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 6. Rota Tiplerinin Dağılımı (Pie Chart)
plt.figure(figsize=(8, 8))
route_counts = df['Route_Type'].value_counts()
plt.pie(route_counts.values, labels=route_counts.index, autopct='%1.1f%%', startangle=140, colors=colors)
plt.title('Rota Tiplerinin Dağılımı')
plt.show()

# 7. Çevre Dostu Durumuna Göre Karbon Salınımı (Boxplot)
plt.figure(figsize=(8, 6))
df.boxplot(column='Carbon_Emission_kgCO2e', by='Is_Eco_Friendly', grid=False)
plt.title('Çevre Dostu Durumuna Göre Karbon Salınımı')
plt.suptitle('') # Otomatik başlığı kaldır
plt.xlabel('Eco-Friendly (0: Hayır, 1: Evet)')
plt.ylabel('Karbon Salınımı (kgCO2e)')
plt.show()





#kısım2

# ============================
# 8. MODELING (2. KISIM)
# ============================

import numpy as np

print("\n================ MODELING BAŞLIYOR ================\n")

# ----------------------------
# 1. MODEL SEÇİMİ
# ----------------------------
# Basit Linear Regression (manuel)
# Amaç: Distance_KM → Carbon_Emission tahmini

# ----------------------------
# 2. DATA PREPARATION
# ----------------------------

# Feature ve target seçimi
X = df[['Distance_KM']].values
y = df['Carbon_Emission_kgCO2e'].values

# Train-Test Split (%80 - %20)
split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Normalization (Min-Max)
X_min = X_train.min()
X_max = X_train.max()

X_train_norm = (X_train - X_min) / (X_max - X_min)
X_test_norm = (X_test - X_min) / (X_max - X_min)

# ----------------------------
# 3. MODEL TRAINING
# ----------------------------

# Linear Regression (Normal Equation)
# w = (X^T X)^-1 X^T y

# Bias ekleme (1'ler sütunu)
X_b = np.c_[np.ones((len(X_train_norm), 1)), X_train_norm]

# Normal equation
theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)

print("Model Parametreleri (theta):")
print(theta)

# ----------------------------
# 4. TAHMİN
# ----------------------------

# Test setine bias ekle
X_test_b = np.c_[np.ones((len(X_test_norm), 1)), X_test_norm]

y_pred = X_test_b.dot(theta)

# ----------------------------
# 5. MODEL EVALUATION
# ----------------------------

# RMSE
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

# MAE
mae = np.mean(np.abs(y_test - y_pred))

# R^2 Score
ss_total = np.sum((y_test - np.mean(y_test))**2)
ss_residual = np.sum((y_test - y_pred)**2)
r2 = 1 - (ss_residual / ss_total)

print("\n--- Model Performansı ---")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R^2: {r2:.4f}")

# ----------------------------
# 6. GÖRSELLEŞTİRME
# ----------------------------

plt.figure(figsize=(10,6))
plt.scatter(X_test, y_test, label="Gerçek Değerler", alpha=0.6)
plt.scatter(X_test, y_pred, label="Tahminler", alpha=0.6)
plt.title("Gerçek vs Tahmin Karbon Salınımı")
plt.xlabel("Distance (KM)")
plt.ylabel("Carbon Emission")
plt.legend()
plt.show()

# Hata grafiği
errors = y_test - y_pred

plt.figure(figsize=(10,6))
plt.hist(errors, bins=30)
plt.title("Hata Dağılımı")
plt.xlabel("Hata")
plt.ylabel("Frekans")
plt.show()

# ----------------------------
# 7. YORUM (PRINT)
# ----------------------------

print("\n--- Yorum ---")
print("Model, mesafe ile karbon salınımı arasında pozitif ilişki olduğunu göstermektedir.")
print("R^2 değeri modelin açıklayıcılığını gösterir.")
print("Düşük RMSE → model iyi tahmin yapıyor demektir.")
print("Bu model SDG kapsamında lojistikte karbon azaltımını analiz etmek için kullanılabilir.")

print("\n--- Limitasyonlar ---")
print("- Sadece tek değişken kullanıldı (Distance)")
print("- Daha iyi model için Vehicle_Type gibi değişkenler eklenebilir")
print("- Non-linear modeller denenebilir")




