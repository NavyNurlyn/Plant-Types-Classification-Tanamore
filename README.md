# 🌿 Klasifikasi Jenis Tanaman dengan InceptionV3

Proyek ini merupakan implementasi model *deep learning* untuk melakukan klasifikasi jenis tanaman (plant species) berdasarkan citra tanaman menggunakan pendekatan *Transfer Learning* dengan arsitektur **InceptionV3**. Proyek ini dikembangkan sebagai bagian dari Capstone Project (C242-PS328).

## 📋 Daftar Isi

- [Deskripsi Proyek](#-deskripsi-proyek)
- [Dataset](#-dataset)
- [Arsitektur Model](#-arsitektur-model)
- [Struktur Notebook](#-struktur-notebook)
- [Instalasi & Requirements](#-instalasi--requirements)
- [Cara Menjalankan](#-cara-menjalankan)
- [Hasil & Evaluasi](#-hasil--evaluasi)
- [Prediksi Gambar Baru](#-prediksi-gambar-baru)
- [Struktur Proyek](#-struktur-proyek)
- [Kontributor](#-kontributor)
- [Lisensi](#-lisensi)

## 📖 Deskripsi Proyek

Model ini dibangun untuk mengenali dan mengklasifikasikan berbagai jenis tanaman rumahan (*house plant species*) dari gambar. Pendekatan yang digunakan adalah *Transfer Learning* dengan memanfaatkan model **InceptionV3** yang telah dilatih sebelumnya (*pre-trained*) pada dataset ImageNet, kemudian dilakukan *fine-tuning* dengan menambahkan *custom classifier layer* untuk menyesuaikan dengan jumlah kelas tanaman pada dataset.

## 📁 Dataset

Dataset yang digunakan berasal dari Kaggle:

🔗 **[House Plant Species](https://www.kaggle.com/datasets/kacpergregorowicz/house-plant-species)** oleh Kacper Gregorowicz

- **Format**: Gambar dikelompokkan dalam folder per kelas (satu folder = satu jenis tanaman)
- **Pembagian data**: 80% *training*, 20% *validation* (menggunakan `validation_split`)
- **Preprocessing**:
  - Resize gambar ke ukuran `224x224` piksel
  - Normalisasi piksel (`rescale=1./255`)
- **Augmentasi data** (untuk mengurangi *overfitting*):
  - Rotasi hingga 40°
  - Pergeseran lebar & tinggi hingga 20%
  - *Shear* dan *zoom* hingga 20%
  - *Horizontal flip*

### Cara mendapatkan dataset

Karena notebook ini awalnya dijalankan dengan dataset yang diunggah ke Google Drive pribadi, path dataset (`/content/drive/My Drive/...`) **kemungkinan besar berbeda** di komputer/akun kamu. Pilih salah satu cara berikut sesuai kebutuhanmu:

**Opsi 1 — Download manual dari Kaggle lalu upload ke Google Drive (paling mudah, cocok untuk Colab)**
1. Download dataset dari link Kaggle di atas.
2. Upload ke Google Drive kamu, misalnya ke folder `MyDrive/Dataset/house_plant_species`.
3. Sesuaikan path ekstraksi di notebook:
   ```python
   with zipfile.ZipFile('/content/drive/My Drive/Dataset/house_plant_species.zip', 'r') as zip_ref:
       zip_ref.extractall('/content/dataset')
   ```

**Opsi 2 — Download langsung via Kaggle API (tidak perlu mount Drive)**
```bash
pip install kaggle
# Pastikan file kaggle.json (API token) sudah ada di ~/.kaggle/ atau diupload ke Colab
kaggle datasets download -d kacpergregorowicz/house-plant-species -p /content/dataset --unzip
```
Lalu sesuaikan `dataset_dir` di notebook agar menunjuk ke folder hasil download, contoh:
```python
dataset_dir = '/content/dataset/house_plant_species'
```

> ⚠️ **Catatan**: Sesuaikan juga nama folder kelas dan `dataset_dir` di notebook dengan struktur folder dataset yang kamu download, karena penamaan folder bisa sedikit berbeda tergantung versi dataset di Kaggle.

## 🧠 Arsitektur Model

| Komponen | Detail |
|---|---|
| Base Model | InceptionV3 (pre-trained ImageNet, tanpa top layer) |
| Base Model Trainable | `False` (freeze seluruh layer) |
| Pooling | GlobalAveragePooling2D |
| Dense Layer | 128 unit, aktivasi ReLU |
| Dropout | 0.5 |
| Output Layer | Dense, aktivasi Softmax (jumlah unit = jumlah kelas) |
| Optimizer | Adam (`learning_rate=0.0001`) |
| Loss Function | Sparse Categorical Crossentropy |
| Metrics | Accuracy |
| Epochs | 20 |
| Batch Size | 32 |
| Input Size | 224 x 224 x 3 |

## 📓 Struktur Notebook

Notebook `Model_2_Klasifikasi_Jenis_Tanaman.ipynb` terdiri dari tahapan berikut:

1. **Import Library** — TensorFlow, Keras, OpenCV, Matplotlib, Scikit-learn, dll.
2. **Mount Google Drive & Ekstraksi Dataset**
3. **Eksplorasi Dataset** — Menghitung jumlah gambar per kelas & visualisasi distribusi
4. **Sample Gambar** — Menampilkan contoh gambar dari tiap kelas
5. **Pengecekan Konsistensi Ukuran Gambar**
6. **Augmentasi Data** — Menggunakan `ImageDataGenerator`
7. **Mapping Label & Indeks Kelas**
8. **Load Model InceptionV3** — Transfer learning + custom classifier
9. **Training Model** — Dengan `ModelCheckpoint` untuk menyimpan model terbaik
10. **Evaluasi Model** — Menghitung akurasi pada data validasi
11. **Visualisasi Hasil Training** — Grafik akurasi & loss (training vs validation)
12. **Prediksi Gambar Baru** — Fungsi untuk memprediksi kelas dari gambar baru

## ⚙️ Instalasi & Requirements

Notebook ini dijalankan di **Google Colab** (dengan akselerator TPU/GPU). Library yang dibutuhkan:

```bash
pip install tensorflow numpy matplotlib opencv-python scikit-learn pillow
```

Library utama yang digunakan:
- `tensorflow` (Keras API)
- `numpy`
- `matplotlib`
- `opencv-python` (cv2)
- `scikit-learn`
- `Pillow` (PIL)

## 🚀 Cara Menjalankan

1. **Buka notebook** di Google Colab.
2. **Siapkan dataset** dengan salah satu opsi pada bagian [Dataset](#-dataset) di atas (download dari Kaggle, lalu upload ke Drive atau pakai Kaggle API).
3. **Mount Google Drive** (jika memakai Opsi 1):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
4. **Sesuaikan path dataset** (`dataset_dir`) di notebook dengan lokasi dataset kamu.
5. **Jalankan seluruh cell secara berurutan** dari atas ke bawah.
6. Model terbaik akan otomatis tersimpan sebagai `inceptionv3_best_model_sparse.h5` selama proses training (berdasarkan `val_accuracy` terbaik).

## 📊 Hasil & Evaluasi

Setelah training selesai, model dievaluasi menggunakan data validasi:

```python
best_model = tf.keras.models.load_model("inceptionv3_best_model_sparse.h5")
val_loss, val_accuracy = best_model.evaluate(validation_dataset)
print(f"Akurasi Validasi: {val_accuracy * 100:.2f}%")
```

Hasil training juga divisualisasikan dalam bentuk grafik **akurasi** dan **loss** (training vs validation) untuk memantau performa dan mendeteksi *overfitting*.

> 💡 Tambahkan hasil akurasi akhir model kamu di sini setelah training selesai, misalnya:
> - **Akurasi Validasi**: `xx.xx%`
> - **Loss Validasi**: `x.xxxx`

## 🔍 Prediksi Gambar Baru

Untuk memprediksi jenis tanaman dari gambar baru, gunakan fungsi `predict_image()`:

```python
def predict_image(img_path, model, class_indices):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class = list(class_indices.keys())[predicted_class_index]

    return predicted_class

predicted_class = predict_image(img_path, best_model, class_indices)
print(f"Hasil Prediksi: {predicted_class}")
```

Fungsi ini akan:
1. Melakukan *preprocessing* gambar (resize ke `224x224` & normalisasi)
2. Melakukan prediksi menggunakan model
3. Mengembalikan label kelas dengan probabilitas tertinggi

## 📂 Struktur Proyek

```
├── Model_2_Klasifikasi_Jenis_Tanaman.ipynb   # Notebook utama
├── inceptionv3_best_model_sparse.h5          # Model terbaik hasil training (output)
└── README.md                                 # Dokumentasi proyek
```

## 👥 Kontributor

Capstone Project — Kelompok C242-PS328

## 📄 Lisensi

Proyek ini dibuat untuk keperluan akademik/Capstone Project. Dataset yang digunakan mengikuti lisensi yang berlaku pada halaman Kaggle: [House Plant Species](https://www.kaggle.com/datasets/kacpergregorowicz/house-plant-species).
