import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def jalankan_simulasi_riset(level_noise):
    np.random.seed(42)
    n_samples = 1000

    # Sinyal BPSK Ideal
    f1_type0 = np.random.normal(loc=1.0, scale=0.2, size=(n_samples // 2, 1))
    f2_type0 = np.random.normal(loc=2.0, scale=0.3, size=(n_samples // 2, 1))
    
    # Sinyal QPSK Ideal
    f1_type1 = np.random.normal(loc=2.5, scale=0.3, size=(n_samples // 2, 1))
    f2_type1 = np.random.normal(loc=3.5, scale=0.4, size=(n_samples // 2, 1))

    X_ideal = np.vstack((np.hstack((f1_type0, f2_type0)), np.hstack((f1_type1, f2_type1))))
    y = np.concatenate((np.zeros(n_samples // 2), np.ones(n_samples // 2)))

    # MAFT-ONN MIT Concept: Menambahkan White Gaussian Noise (AWGN) ke dalam sinyal
    noise = np.random.normal(loc=0.0, scale=level_noise, size=X_ideal.shape)
    X_berderau = X_ideal + noise

    # Proses Machine Learning
    X_train, X_test, y_train, y_test = train_test_split(X_berderau, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return accuracy_score(y_test, model.predict(X_test))

# Eksperimen 1: Kondisi Lab / Sinyal Sempurna
akurasi_bersih = jalankan_simulasi_riset(level_noise=0.0)
print(f"[EKSPERIMEN 1] Akurasi pada Sinyal Sempurna (No Noise): {akurasi_bersih * 100:.2f}%")

# Eksperimen 2: Kondisi Cuaca Buruk / Gangguan Spektrum Tinggi
akurasi_buruk = jalankan_simulasi_riset(level_noise=1.2)
print(f"[EKSPERIMEN 2] Akurasi pada Sinyal Gangguan Cuaca Ekstrem: {akurasi_buruk * 100:.2f}%")
