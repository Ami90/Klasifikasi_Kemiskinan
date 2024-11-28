import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import warnings

# Mengatur tampilan
st.set_option('deprecation.showPyplotGlobalUse', False)
warnings.filterwarnings(action='ignore')

# Judul aplikasi
st.title("Klasifikasi Tingkat Kemiskinan di Indonesia")
st.write("Aplikasi untuk melakukan klasifikasi tingkat kemiskinan berdasarkan fitur-fitur tertentu.")

# Load dataset
perth_file_path = './project/Klasifikasi Tingkat Kemiskinan di Indonesia.csv'
ptemp_data = pd.read_csv(perth_file_path, delimiter=';')

# Tampilkan kolom dataset
st.write("Kolom yang terdeteksi dalam dataset:", list(ptemp_data.columns))

# Ubah data ke int
for attribute in ptemp_data:
    ptemp_data[attribute] = LabelEncoder().fit_transform(ptemp_data[attribute])

# Histogram
st.subheader("Histogram Distribusi Data")
fig, ax = plt.subplots(figsize=(15, 8))
ptemp_data.hist(ax=ax)
plt.title('Pesebaran Data')
plt.tight_layout()
st.pyplot(fig)

# Definisi fitur (features) dan target (target)
features = ['Rata-rata Lama Sekolah Penduduk 15+ (Tahun)',
            'Pengeluaran per Kapita Disesuaikan (Ribu Rupiah/Orang/Tahun)',
            'Indeks Pembangunan Manusia',
            'Umur Harapan Hidup (Tahun)',
            'Persentase rumah tangga yang memiliki akses terhadap sanitasi layak',
            'Persentase rumah tangga yang memiliki akses terhadap air minum layak',
            'Tingkat Pengangguran Terbuka',
            'Tingkat Partisipasi Angkatan Kerja',
            'PDRB atas Dasar Harga Konstan menurut Pengeluaran (Rupiah)']

target_column = 'Persentase Penduduk Miskin (P0) Menurut Kabupaten/Kota (Persen)'
target = (ptemp_data[target_column] > 20).astype(int)  # Binary classification: 1 if > 20, 0 otherwise

# Split data menjadi data latih dan data uji
train_X, val_X, train_Y, val_Y = train_test_split(ptemp_data[features], target, random_state=0)

# Inisialisasi dan melatih model Regresi Logistik
df_clear_Logistic_Regression = LogisticRegression(random_state=0).fit(train_X, train_Y)

# Membuat prediksi pada data uji
val_predictions = df_clear_Logistic_Regression.predict(val_X)

# Menghitung akurasi
accuracy = accuracy_score(val_Y, val_predictions)
st.write(f'Akurasi Model: {accuracy:.2f}')

# Menampilkan laporan klasifikasi
st.subheader("Laporan Klasifikasi:")
st.text(classification_report(val_Y, val_predictions))

# Tambahkan opsi untuk melakukan prediksi baru
st.subheader("Prediksi Tingkat Kemiskinan Baru")
input_data = {}
for feature in features:
    input_data[feature] = st.number_input(f"Masukkan nilai untuk {feature}:", value=0.0)

if st.button("Prediksi"):
    input_df = pd.DataFrame([input_data])
    prediction = df_clear_Logistic_Regression.predict(input_df)
    st.write(f"Hasil Prediksi: {'Miskin' if prediction[0] == 1 else 'Tidak Miskin'}")