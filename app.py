import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# --- 1. Muat objek yang telah disimpan --- 
@st.cache_resource
def load_artifacts():
    try:
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        with open('label_encoders.pkl', 'rb') as file:
            label_encoders = pickle.load(file)
        with open('one_hot_columns.pkl', 'rb') as file:
            one_hot_columns = pickle.load(file)
        with open('model_huber.pkl', 'rb') as file:
            model = pickle.load(file)
        return scaler, label_encoders, one_hot_columns, model
    except FileNotFoundError:
        st.error("File model, scaler, atau encoder tidak ditemukan. Pastikan file-file .pkl berada di direktori yang sama.[cite: 1]")
        st.stop()

scaler, label_encoders, one_hot_columns, model = load_artifacts()

# --- 2. Definisikan pemetaan dan daftar kategori ---
mapping_gender_ui_to_processing = {'Laki-laki':'L', 'Perempuan':'P'}[cite: 1]

# Mengambil opsi langsung dari LabelEncoder yang dimuat[cite: 1]
pendidikan_options = list(label_encoders['Pendidikan'].classes_)
jurusan_options = list(label_encoders['Jurusan'].classes_)

# Urutan fitur yang sesuai dengan X_train saat modeling[cite: 1]
expected_model_columns = [
    'Usia', 'Durasi_Jam', 'Nilai_Ujian', 'Pendidikan', 'Jurusan',
    'Jenis_Kelamin_L', 'Jenis_Kelamin_P', 
    'Status_Bekerja_Belum Bekerja', 'Status_Bekerja_Sudah Bekerja'
]

# --- 3. Fungsi Preprocessing (Perbaikan pada bagian Label Encoding) ---
def preprocess_input(input_data):
    # Buat DataFrame dari input tunggal[cite: 1]
    df_input = pd.DataFrame([input_data])

    # a. Standardisasi Jenis Kelamin[cite: 1]
    df_input['Jenis_Kelamin'] = df_input['Jenis_Kelamin'].map(mapping_gender_ui_to_processing)

    # b. Huruf Kecil Jurusan[cite: 1]
    df_input['Jurusan'] = df_input['Jurusan'].str.lower()

    # c. Menghapus Spasi (Strip) pada kolom kategorikal[cite: 1]
    for col in ['Jenis_Kelamin', 'Jurusan', 'Pendidikan', 'Status_Bekerja']:
        if col in df_input.columns:
            df_input[col] = df_input[col].str.strip()

    # d. Label Encoding (DIPERBAIKI untuk menghindari TypeError)
    for col in ['Pendidikan', 'Jurusan']:
        if col in label_encoders:
            # Transformasi data menjadi array numerik
            transformed_values = label_encoders[col].transform(df_input[col])
            # Masukkan kembali ke DataFrame sebagai tipe integer eksplisit
            df_input[col] = transformed_values
            df_input[col] = df_input[col].astype(int)
        else:
            st.warning(f"LabelEncoder untuk kolom '{col}' tidak ditemukan.[cite: 1]")

    # e. One-Hot Encoding untuk 'Jenis_Kelamin' dan 'Status_Bekerja'[cite: 1]
    one_hot_cols_to_process = ['Jenis_Kelamin', 'Status_Bekerja']
    df_onehot_raw = pd.get_dummies(df_input[one_hot_cols_to_process], prefix=one_hot_cols_to_process, dtype=int)

    # Gabungkan fitur numerik/label-encoded dengan one-hot encoded[cite: 1]
    processed_df_temp = df_input.drop(columns=one_hot_cols_to_process).copy()
    processed_df = pd.concat([processed_df_temp, df_onehot_raw], axis=1)

    # Pastikan semua kolom model ada dan urutannya benar[cite: 1]
    final_processed_input = pd.DataFrame(columns=expected_model_columns)
    for col in expected_model_columns:
        if col in processed_df.columns:
            final_processed_input[col] = processed_df[col]
        else:
            final_processed_input[col] = 0 
    
    # Memastikan tipe data konsisten sebelum masuk ke scaler[cite: 1]
    final_processed_input['Pendidikan'] = final_processed_input['Pendidikan'].astype(int)
    final_processed_input['Jurusan'] = final_processed_input['Jurusan'].astype(int)

    # f. Penskalaan menggunakan scaler yang dimuat[cite: 1]
    scaled_input = scaler.transform(final_processed_input)
    scaled_input_df = pd.DataFrame(scaled_input, columns=expected_model_columns)

    return scaled_input_df

# --- 4. Streamlit UI ---
st.set_page_config(page_title="Prediksi Gaji Lulusan", layout="centered")
st.title("Aplikasi Prediksi Gaji Awal Lulusan Pelatihan Vokasi[cite: 1]")
st.write("Isi formulir di bawah untuk memprediksi gaji awal (dalam Juta Rupiah).[cite: 1]")

with st.form("prediction_form"):
    st.header("Data Personal")
    usia = st.slider("Usia (tahun)", min_value=17, max_value=45, value=25, step=1)[cite: 1]
    jenis_kelamin = st.selectbox("Jenis Kelamin", options=["Laki-laki", "Perempuan"])[cite: 1]
    pendidikan = st.selectbox("Pendidikan", options=pendidikan_options)[cite: 1]
    status_bekerja = st.selectbox("Status Bekerja", options=["Sudah Bekerja", "Belum Bekerja"])[cite: 1]

    st.header("Data Pelatihan")
    jurusan = st.selectbox("Jurusan Pelatihan", options=jurusan_options)[cite: 1]
    durasi_jam = st.slider("Durasi Pelatihan (Jam)", min_value=37, max_value=89, value=60, step=1)[cite: 1]
    nilai_ujian = st.slider("Nilai Ujian", min_value=71, max_value=100, value=85, step=1)[cite: 1]

    submit_button = st.form_submit_button("Prediksi Gaji")[cite: 1]

    if submit_button:
        input_data = {
            'Usia': usia,
            'Durasi_Jam': durasi_jam,
            'Nilai_Ujian': nilai_ujian,
            'Pendidikan': pendidikan,
            'Jurusan': jurusan,
            'Jenis_Kelamin': jenis_kelamin,
            'Status_Bekerja': status_bekerja
        }[cite: 1]

        # Jalankan preprocessing dan prediksi[cite: 1]
        processed_input = preprocess_input(input_data)
        predicted_gaji = model.predict(processed_input)[0]

        st.subheader("Hasil Prediksi")
        st.success(f"Prediksi Gaji Awal: **Rp {predicted_gaji:.2f} Juta Rupiah**[cite: 1]")
