# Klasifikasi Keterbacaan Plat Nomor (Readable vs Unreadable)

Proyek ini bertujuan untuk mengklasifikasikan gambar potongan plat nomor kendaraan ke dalam dua kategori: **Readable (Dapat dibaca)** dan **Unreadable (Tidak dapat dibaca)**. Proyek ini menggunakan beberapa pendekatan Machine Learning dan Deep Learning untuk menemukan model dengan performa terbaik pada klasifikasi gambar.

---

## Ringkasan Proyek

Pendeteksian keterbacaan plat nomor merupakan salah satu komponen penting dalam sistem _Automatic Number Plate Recognition_ (ANPR). Dalam proyek ini, dataset gambar plat nomor diekstraksi dan dilatih menggunakan tiga pendekatan model yang berbeda untuk membandingkan kinerjanya.

**Distribusi Dataset:**

- **Training:** 1111 sampel
- **Validation:** 238 sampel
- **Testing:** 239 sampel
- _Catatan:_ Dataset telah diaugmentasi untuk menangani _imbalance class_ (ketidakseimbangan jumlah data antara kelas _readable_ dan _unreadable_).

---

## Cara Menjalankan Proyek

Proyek ini dibuat dan dilatih di lingkungan berbasis Jupyter Notebook / Google Colab (`preprocessing_modelling_UAS.ipynb`). Berikut adalah langkah-langkah untuk menjalankannya:

1. **Siapkan Lingkungan Eksekusi:**
   Sangat disarankan untuk menjalankan notebook ini di Google Colab karena membutuhkan komputasi GPU untuk pelatihan model _Deep Learning_.
2. **Siapkan Dataset:**
   Pastikan direktori dataset sesuai dengan _path_ yang ada di dalam _notebook_ (default menggunakan path Google Drive `/content/drive/MyDrive/...`).
3. **Instalasi Dependensi:**
   Di dalam _notebook_ sudah terdapat _cell_ untuk menginstal _library_ yang dibutuhkan. Beberapa pustaka utama meliputi:
   ```bash
   pip install tensorflow albumentations scikit-image scikit-learn opencv-python-headless pandas seaborn matplotlib
   ```
4. **Jalankan Cell secara Berurutan:**
   - Jalankan proses Mounting Google Drive (jika menggunakan Colab)
   - Jalankan bagian Preprocessing & Feature Extraction.
   - Jalankan tahap pelatihan dan evaluasi masing-masing model.
5. **Menyimpan Model:**
   _Notebook_ ini akan secara otomatis menyimpan model yang telah dilatih beserta _scaler_ dalam format `.pkl` (untuk SVM) dan `.keras` (untuk MobileNetV3) ke direktori yang ditentukan.

---

## Hasil Evaluasi Model

Proyek ini membandingkan 3 pendekatan model klasifikasi. Berikut adalah ringkasan hasil evaluasi performa dari setiap model pada **Data Testing**:

### 1. Model SVM Murni (HOG + LBP)

Model ini menggunakan teknik ekstraksi fitur tradisional yaitu _Histogram of Oriented Gradients_ (HOG) yang digabung dengan _Local Binary Pattern_ (LBP) sebelum diklasifikasikan dengan Support Vector Machine (SVM).

- **Akurasi Validation:** 95.38%
- **Akurasi Testing:** **94.14%**
- **Detail:** Menggunakan hyperparameter tuning _Grid Search_ untuk mendapatkan parameter terbaik (`C=1`, `gamma='scale'`, `kernel='rbf'`). Ini adalah model dengan **performa keseluruhan terbaik** pada dataset test.

### 2. Model MobileNetV3 (Fine-tuning)

Pendekatan _Deep Learning_ menggunakan arsitektur MobileNetV3-Small yang telah dilatih sebelumnya (Transfer Learning) dan di-_fine-tune_ menggunakan _learning rate_ kecil.

- **Akurasi Validation:** 91.60%
- **Akurasi Testing:** **91.63%**
- **Detail:** Walaupun akurasinya cukup baik, model cenderung _overfit_ lebih cepat pada epoch awal dan membutuhkan kalibrasi probabilitas _threshold_ secara hati-hati untuk kelas minoritas (_unreadable_).

### 3. Model Hybrid (MobileNetV3 + SVM)

Model ini menggabungkan MobileNetV3 sebagai **Feature Extractor** (menghasilkan representasi vektor 576 dimensi) dan menukarkan klasifikasi lapisan akhir menggunakan **SVM**.

- **Akurasi Validation:** 94.00%
- **Akurasi Testing:** **91.21%**
- **Detail:** Performa Hybrid ini sedikit di bawah SVM Murni, kemungkinan karena SVM kesulitan memisahkan _feature map_ mentah dari MobileNetV3 jika dibandingkan dengan fitur _handcrafted_ seperti HOG+LBP untuk kasus dataset beresolusi spesifik ini.

---

## Kesimpulan

Dari eksperimen yang dilakukan dalam proyek ini, **SVM Murni dengan ekstraksi fitur HOG+LBP** terbukti lebih tangguh dan akurat secara komputasi untuk mengklasifikasikan plat nomor kendaraan yang _readable_ vs _unreadable_, disusul oleh model berbasis arsitektur _MobileNetV3_.
