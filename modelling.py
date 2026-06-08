import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Load Dataset Hasil Preprocessing
df = pd.read_csv('rekammedis_preprocessing.csv')

# 2. Memisahkan Target (Test Results) dan Fitur
X = df.drop('Test Results', axis=1)
y = df['Test Results']

# Mengubah data teks kategori (Gender, Blood Type, dll) menjadi angka agar bisa diproses model
X = pd.get_dummies(X, drop_first=True)

# Membagi data training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Konfigurasi MLflow Lokal (Wajib diarahkan ke localhost untuk tingkat Basic)
mlflow.set_tracking_uri("http://127.0.0.1:5001")
mlflow.set_experiment("Eksperimen_Rekam_Medis")

# Mengaktifkan autolog untuk mencatat metrik secara otomatis
mlflow.autolog()

# 4. Melatih Model
with mlflow.start_run():
    # Model Random Forest sederhana tanpa hyperparameter tuning
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    print("Training berhasil! Silakan cek MLflow UI Anda.")