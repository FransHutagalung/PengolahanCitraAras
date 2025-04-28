# Aplikasi Pembelajaran Pengolahan Citra Digital

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-orange)](https://opencv.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0%2B-lightgrey)](https://flask.palletsprojects.com/)

Aplikasi web interaktif untuk mempelajari teknik pengolahan citra digital pada berbagai aras pemrosesan.

## 📑 Daftar Isi
- [Fitur](#-fitur)
- [Instalasi](#-instalasi)
- [Penggunaan](#-penggunaan)
- [Struktur Kode](#-struktur-kode)
- [API Endpoints](#-api-endpoints)
- [Kontribusi](#-kontribusi)
- [Lisensi](#-lisensi)

## 🌟 Fitur

### Aras Pemrosesan Citra
1. **Aras Titik (Point Processing)**
   - Brightness Adjustment
   - Contrast Adjustment
   - Thresholding
   - Image Negation
   - Histogram Equalization

2. **Aras Lokal (Local Processing)**
   - Gaussian Blur
   - Median Filter
   - Image Sharpening
   - Edge Detection (Sobel)
   - Adaptive Thresholding

3. **Aras Global (Global Processing)**
   - Fourier Transform
   - Low-Pass Filter
   - High-Pass Filter
   - Histogram Matching

4. **Aras Objek (Object Processing)**
   - K-Means Segmentation
   - Connected Components
   - Watershed Segmentation
   - Contour Detection

## 📥 Instalasi

### Persyaratan Sistem
- Python 3.8+
- pip package manager

### Langkah Instalasi
1. Clone repository
```bash
git clone https://github.com/username/pengolahan-citra-aras.git
cd pengolahan-citra-aras
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Jalankan aplikasi
```bash
python app.py
```

Aplikasi akan berjalan di `http://localhost:5000`

## 🖥️ Penggunaan

1. **Antarmuka Web**
   - Unggah gambar melalui tombol upload
   - Pilih jenis pemrosesan dari menu dropdown
   - Atur parameter sesuai kebutuhan
   - Lihat hasil pemrosesan dan penjelasan algoritma

2. **Demo Semua Fitur**
   - Navigasi ke `/demo_semua` untuk melihat demo lengkap
   - Pilih kategori pemrosesan untuk melihat perbandingan teknik

## 🧠 Struktur Kode

```bash
.
├── app.py                 # Main application (Flask)
├── PembelajaranPengolahanCitra.py # Core image processing class
├── templates/
│   └── index.html         # Web interface
├── static/                # CSS/JS assets
├── requirements.txt       # Dependencies
└── README.md
```

## 🌐 API Endpoints

| Endpoint | Method | Deskripsi |
|----------|--------|-----------|
| `/upload` | POST | Upload gambar |
| `/proses/<jenis_aras>/<teknik>` | POST | Proses gambar dengan teknik tertentu |
| `/demo/<jenis_aras>` | GET | Dapatkan demo untuk kategori tertentu |
| `/demo_semua` | GET | Dapatkan demo semua kategori |

Contoh request:
```bash
curl -X POST -F "file=@input.jpg" http://localhost:5000/upload
```

## 📚 Dependensi Utama

- OpenCV (`cv2`) - Operasi dasar pengolahan citra
- scikit-image (`skimage`) - Segmentasi dan analisis gambar
- scikit-learn (`sklearn`) - Algoritma K-Means
- Flask - Web framework
- NumPy - Operasi array
- Matplotlib - Visualisasi

## 🤝 Kontribusi

Kontribusi terbuka! Ikuti langkah berikut:
1. Fork repository
2. Buat branch fitur (`git checkout -b fitur/namafitur`)
3. Commit perubahan (`git commit -m 'Tambahkan fitur x'`)
4. Push ke branch (`git push origin fitur/namafitur`)
5. Buat Pull Request

## 📜 Lisensi

Distributed under the MIT License. Lihat `LICENSE` untuk detail lebih lanjut.