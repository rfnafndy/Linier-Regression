import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

# Fungsi untuk memuat data
@st.cache_data
def load_data(file):
    data = pd.read_csv(file)
    return data

# Fungsi untuk melakukan prediksi dengan regresi linier sederhana
def predict_sales(data, feature_col, target_col):
    X = data[[feature_col]].values
    y = data[target_col].values
    
    # Normalisasi data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Latih model
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    # Prediksi nilai target (penjualan)
    y_pred = model.predict(X_scaled)
    
    # Hitung MAE dan MAPE
    mae = mean_absolute_error(y, y_pred)
    mape = mean_absolute_percentage_error(y, y_pred)
    
    # Ambil slope dan intercept
    slope = model.coef_[0]
    intercept = model.intercept_
    
    return y, y_pred, mae, mape, model, scaler, slope, intercept

# Fungsi untuk menyimpan prediksi dalam session state
def save_predictions(predictions):
    st.session_state['predictions'] = predictions

# Halaman untuk mengunggah file
def upload_page():
    st.title("Upload File CSV")
    uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])
    
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        st.write("Data yang diunggah:")
        st.write(data.head())
        st.session_state['data'] = data

    if st.button("Lanjutkan ke Prediksi"):
        st.session_state.page = "Prediksi Data"
        st.experimental_rerun()

# Halaman untuk melakukan prediksi
def predict_page():
    st.title("Prediksi Data")
    
    if 'data' in st.session_state:
        data = st.session_state['data']
        st.write("Data yang akan digunakan untuk prediksi:")
        st.write(data.head())
        
        st.subheader("Konfigurasi Regresi")
        feature_col = st.selectbox("Pilih fitur untuk prediksi (variabel independen)", data.columns)
        target_col = st.selectbox("Pilih kolom target penjualan (variabel dependen)", data.columns)
        
        # Memeriksa apakah kolom yang dipilih adalah kolom numerik
        if not pd.api.types.is_numeric_dtype(data[feature_col]):
            st.write("Kolom fitur harus berupa tipe data numerik.")
            return
        
        if not pd.api.types.is_numeric_dtype(data[target_col]):
            st.write("Kolom target harus berupa tipe data numerik.")
            return
        
        if st.button("Mulai Prediksi"):
            st.write("Proses prediksi sedang berlangsung...")
            y, y_pred, mae, mape, model, scaler, slope, intercept = predict_sales(data, feature_col, target_col)
            
            # Simpan hasil prediksi dalam session state
            save_predictions((y, y_pred, mae, mape, model, scaler, slope, intercept, feature_col, target_col))
            
            st.write("Prediksi berhasil dilakukan.")
            st.write(f"Slope (Kemiringan): {slope}")
            st.write(f"Intercept (Titik Potong): {intercept}")
            st.write(f"Mean Absolute Error (MAE): {mae}")
            st.write(f"Mean Absolute Percentage Error (MAPE): {mape}%")
    
    else:
        st.write("Silakan unggah file data terlebih dahulu pada halaman Upload File CSV.")
    
    if st.button("Lanjutkan ke Visualisasi"):
        st.session_state.page = "Visualisasi Hasil"
        st.experimental_rerun()

# Halaman untuk visualisasi hasil
def visualize_page():
    st.title("Visualisasi Hasil Prediksi")
    
    if 'predictions' in st.session_state:
        y, y_pred, mae, mape, model, scaler, slope, intercept, feature_col, target_col = st.session_state['predictions']
        data = st.session_state['data']
        
        st.write("Hasil Prediksi:")
        result_df = pd.DataFrame({'Actual': y, 'Predicted': y_pred})
        st.write(result_df)
        
        st.subheader("Visualisasi Hasil Prediksi")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(data[feature_col], data[target_col], color='blue', label='Actual')
        ax.plot(data[feature_col], y_pred, color='red', label='Prediction')
        ax.set_xlabel(feature_col)
        ax.set_ylabel(target_col)
        ax.set_title(f'Regresi Linier Sederhana: Prediksi vs Aktual')
        ax.legend()
        st.pyplot(fig)
    
    else:
        st.write("Silakan lakukan prediksi terlebih dahulu pada halaman Prediksi Data.")

# Main function
def main():
    if 'page' not in st.session_state:
        st.session_state.page = "Upload File CSV"

    if st.session_state.page == "Upload File CSV":
        upload_page()
    elif st.session_state.page == "Prediksi Data":
        predict_page()
    elif st.session_state.page == "Visualisasi Hasil":
        visualize_page()

    st.sidebar.title("Navigasi")
    if st.sidebar.button("Upload File CSV"):
        st.session_state.page = "Upload File CSV"
        st.experimental_rerun()
    if st.sidebar.button("Prediksi Data"):
        st.session_state.page = "Prediksi Data"
        st.experimental_rerun()
    if st.sidebar.button("Visualisasi Hasil"):
        st.session_state.page = "Visualisasi Hasil"
        st.experimental_rerun()

# Panggil fungsi utama untuk menjalankan aplikasi
if __name__ == "__main__":
    main()
