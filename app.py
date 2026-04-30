import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.preprocessing import LabelEncoder

# --- 1. Muat objek yang telah disimpan --- 
@st.cache_resource
def load_artifacts():
    # Daftar file yang dibutuhkan
    files = ['scaler.pkl', 'label_encoders.pkl', 'one_hot_columns.pkl', 'model_huber.pkl']
    
    # Cek apakah semua file ada di direktori
    for f in files:
        if not os.path.exists(f):
            st.error(f"File **{f}** tidak ditemukan! Pastikan file ini sudah diunggah ke GitHub/direktori yang sama.")
            st.stop()

    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    with open('label_encoders.pkl', 'rb') as file:
        label_encoders = pickle.load(file)
    with open('one_hot_columns.pkl', 'rb') as file:
        one_hot_columns = pickle.load(file)
    with open('model_huber.pkl', 'rb') as file:
        model = pickle.load(file)
        
    return scaler, label_encoders, one_hot_columns, model

scaler, label_encoders, one_hot_columns, model = load_artifacts()

# --- 2. Konfigurasi UI & Mapping ---
mapping_gender_ui_to_processing = {'Laki-laki':'L', 'Perempuan':'P'}

# Ambil list kategori untuk selectbox agar sesuai dengan saat training
pendidikan_options = list(label_encoders['Pendidikan'].classes_)
jurusan_options = list(label_encoders['Jurusan'].classes_)

# Urutan kolom yang WAJIB sesuai dengan saat model dilatih (X_train)
expected_model_columns = [
    'Usia', 'Durasi_Jam', 'Nilai_Ujian', 'Pendidikan', 'Jurusan',
    'Jenis_Kelamin_L', 'Jenis_Kelamin_P', 
    'Status_Bekerja_Belum Bekerja', 'Status_Bekerja_Sudah Bekerja'
]

# --- 3. Fungsi Preprocessing ---
def preprocess_input(input_data):
    # Buat DataFrame awal
    df_input = pd.DataFrame([input_data])

    # a. Mapping Gender (Laki-laki -> L, Perempuan -> P)
    df_input['Jenis_Kelamin'] = df_input['Jenis_Kelamin'].map(mapping_gender_ui_to_processing)

    # b. Membersihkan teks (Huruf kecil & hapus spasi)
    df_input['Jurusan'] = df_input['Jurusan'].str.lower().str.strip()
    df_input['Pendidikan'] = df_input['Pendidikan'].str.strip()
    df_input['Status_Bekerja'] = df_input['Status_Bekerja'].str.strip()

    # c. Label Encoding (Perbaikan krusial: hindari TypeError dengan assignment langsung)
    for col in ['Pendidikan', 'Jurusan']:
        # Melakukan transformasi dan memastikan tipe data menjadi integer
        df_input[col] = label_encoders[col].transform(df_input[col])
        df_input[col] = df_input[col].astype(int)

    # d. One-Hot Encoding
    # Kita buat kolom dummy untuk Jenis_Kelamin dan Status_Bekerja
    df_onehot = pd.get_dummies(df_input[['Jenis_Kelamin', 'Status_Bekerja']], dtype=int)

    # e. Menggabungkan kolom numerik & label encoded dengan hasil one-hot
    df_final = pd.concat([df_input[['Usia', 'Durasi_Jam', 'Nilai_Ujian', 'Pendidikan', 'Jurusan']], df_onehot], axis=1)

    # f. Sinkronisasi Kolom (Pastikan semua kolom expected_model_columns ada)
    # Jika ada kolom yang kurang (misal user hanya pilih satu kategori), tambahkan dengan nilai 0
    for col in expected_model_columns:
        if col not in df_final.columns:
            df_final[col] = 0

    # Pastikan urutan kolom sesuai urutan model
    df_final = df_final[expected_model_columns]

    # g. Feature Scaling
    scaled_data = scaler.transform(df_final)
    
    return scaled_data

# --- 4. Tampilan Streamlit ---
st.set_page_config(page_title="Prediksi Gaji Vokasi", layout="centered")

st.title("💰 Aplikasi Prediksi Gaji Awal")
st.markdown("---")

with st.form("my_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        usia = st.number_input("Usia", min_value=15, max_value=60, value=25)
        jenis_kelamin = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
        pendidikan = st.selectbox("Tingkat Pendidikan", pendidikan_options)
        
    with col2:
        status_bekerja = st.selectbox("Status Saat Ini", ["Belum Bekerja", "Sudah Bekerja"])
        jurusan = st.selectbox("Jurusan Pelatihan", jurusan_options)
        durasi_jam = st.number_input("Durasi Pelatihan (Jam)", min_value=1, max_value=500, value=40)
        nilai_ujian = st.slider("Nilai Ujian", 0, 100, 85)

    submit = st.form_submit_button("Hitung Prediksi Gaji")

if submit:
    # Data input dari form
    data = {
        'Usia': usia,
        'Durasi_Jam': durasi_jam,
        'Nilai_Ujian': nilai_ujian,
        'Pendidikan': pendidikan,
        'Jurusan': jurusan,
        'Jenis_Kelamin': jenis_kelamin,
        'Status_Bekerja': status_bekerja
    }
    
    try:
        # Proses dan Prediksi
        features = preprocess_input(data)
        prediction = model.predict(features)[0]
        
        st.markdown("---")
        st.subheader("Hasil Estimasi:")
        # Menampilkan hasil dalam format Rupiah (asumsi unit di model adalah Juta)
        st.success(f"Estimasi Gaji Awal Anda: **Rp {prediction:.2f} Juta**")
        
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses data: {e}")
