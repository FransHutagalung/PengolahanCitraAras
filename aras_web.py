"""
APLIKASI PEMBELAJARAN PENGOLAHAN CITRA DIGITAL
==============================================
Aplikasi ini memperlihatkan berbagai teknik pengolahan citra digital
dengan pendekatan aras (level) yang berbeda:
- Aras Titik: Operasi yang memproses piksel secara individual
- Aras Lokal: Operasi yang memproses piksel berdasarkan tetangganya
- Aras Global: Operasi yang memproses seluruh gambar
- Aras Objek: Operasi yang memproses objek dalam gambar
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import base64
import io
from skimage import feature, measure, segmentation
from sklearn.cluster import KMeans
from scipy import ndimage
from flask import Flask, render_template, request, redirect, url_for, jsonify

# Konfigurasi matplotlib untuk menyimpan plot sebagai gambar untuk web
plt.switch_backend('Agg')

class PembelajaranPengolahanCitra:
    def __init__(self, gambar_input=None, is_path=True):
        """
        Inisialisasi kelas pembelajaran pengolahan citra
        
        Parameters:
        -----------
        gambar_input : str atau numpy.ndarray
            Path gambar atau array gambar yang akan diproses
        is_path : bool
            True jika gambar_input adalah path file, False jika numpy array
        """
        if gambar_input is not None:
            if is_path:
                self.gambar_original = cv2.imread(gambar_input)
                if self.gambar_original is None:
                    raise FileNotFoundError(f"Gambar tidak ditemukan: {gambar_input}")
                self.gambar_path = gambar_input
            else:
                self.gambar_original = gambar_input
                self.gambar_path = None
            
            # Konversi BGR ke RGB untuk matplotlib
            self.gambar_rgb = cv2.cvtColor(self.gambar_original, cv2.COLOR_BGR2RGB)
            
            # Konversi ke grayscale
            self.gambar_gray = cv2.cvtColor(self.gambar_original, cv2.COLOR_BGR2GRAY)
            
            print(f"Gambar berhasil dimuat dengan ukuran: {self.gambar_original.shape}")
            self.langkah_terakhir = "Gambar awal dimuat dan dikonversi ke format RGB dan grayscale"
        else:
            self.gambar_original = None
            self.gambar_rgb = None
            self.gambar_gray = None
            self.gambar_path = None
            self.langkah_terakhir = "Belum ada gambar yang dimuat"
    
    def muat_gambar(self, gambar_path):
        """
        Muat gambar dari path
        
        Parameters:
        -----------
        gambar_path : str
            Path ke file gambar
        """
        self.gambar_original = cv2.imread(gambar_path)
        if self.gambar_original is None:
            raise FileNotFoundError(f"Gambar tidak ditemukan: {gambar_path}")
        
        self.gambar_path = gambar_path
        self.gambar_rgb = cv2.cvtColor(self.gambar_original, cv2.COLOR_BGR2RGB)
        self.gambar_gray = cv2.cvtColor(self.gambar_original, cv2.COLOR_BGR2GRAY)
        
        print(f"Gambar berhasil dimuat dengan ukuran: {self.gambar_original.shape}")
        self.langkah_terakhir = "Gambar baru dimuat dan dikonversi ke format RGB dan grayscale"
        return self
    
    def tampilkan_gambar(self, gambar_list, judul_list, ukuran=(15, 10), tahapan=None, dpi=100):
        """
        Menampilkan beberapa gambar dengan pyplot dan menjelaskan tahapan
        
        Parameters:
        -----------
        gambar_list : list
            Daftar gambar yang akan ditampilkan
        judul_list : list
            Daftar judul untuk setiap gambar
        ukuran : tuple
            Ukuran figure matplotlib
        tahapan : list atau None
            Daftar penjelasan tahapan untuk setiap gambar, jika None maka tidak ditampilkan
        dpi : int
            Dots per inch untuk output gambar
            
        Returns:
        --------
        bytes
            Representasi byte dari gambar yang ditampilkan
        """
        fig, axes = plt.subplots(1, len(gambar_list), figsize=ukuran)
        if len(gambar_list) == 1:
            axes = [axes]
        
        for i, (gambar, judul) in enumerate(zip(gambar_list, judul_list)):
            if len(gambar.shape) == 2:
                axes[i].imshow(gambar, cmap='gray')
            else:
                axes[i].imshow(gambar)
            axes[i].set_title(judul)
            
            # Tambahkan penjelasan tahapan jika disediakan
            if tahapan and i < len(tahapan):
                axes[i].set_xlabel(tahapan[i], fontsize=8, wrap=True)
            
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # Simpan plot ke dalam buffer untuk web
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=dpi)
        buf.seek(0)
        
        # Bersihkan plot untuk menghindari tumpang tindih
        plt.close(fig)
        
        return buf
    
    def gambar_ke_base64(self, buf):
        """
        Konversi buffer gambar ke string base64 untuk HTML
        
        Parameters:
        -----------
        buf : io.BytesIO
            Buffer yang berisi data gambar
            
        Returns:
        --------
        str
            String base64 untuk digunakan di HTML
        """
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    
    # ===== ARAS TITIK =====
    def aras_titik_brightness(self, nilai=50):
        """
        Mengubah kecerahan gambar (operasi titik)
        
        Penjelasan Teknis:
        ------------------
        Operasi aras titik dengan brightness bekerja dengan menambah atau mengurangi
        nilai piksel secara seragam. Setiap piksel dimodifikasi dengan nilai tertentu,
        tanpa melihat piksel tetangganya.
        
        Langkah-langkah:
        1. Konversi gambar ke tipe data int16 untuk menghindari overflow
        2. Tambahkan nilai brightness ke setiap piksel
        3. Clip nilai piksel ke rentang [0,255]
        4. Konversi kembali ke tipe data uint8
        
        Parameters:
        -----------
        nilai : int
            Nilai perubahan kecerahan (-255 sampai 255)
            
        Returns:
        --------
        numpy.ndarray
            Gambar dengan kecerahan yang diubah
        """
        # Simpan langkah proses
        self.langkah_terakhir = f"""
        Proses Brightness Adjustment (Aras Titik):
        1. Konversi gambar ke tipe data int16 untuk menghindari overflow/underflow pada operasi
        2. Setiap piksel ditambah dengan nilai brightness {nilai}
        3. Nilai hasil dibatasi (clipping) antara 0-255 untuk memastikan tetap valid
        4. Konversi kembali ke tipe data uint8 standar untuk gambar
        """
        
        # Proses brightness adjustment
        hasil = np.clip(self.gambar_rgb.astype(np.int16) + nilai, 0, 255).astype(np.uint8)
        return hasil
    
    def aras_titik_contrast(self, alpha=1.5):
        """
        Mengubah kontras gambar (operasi titik)
        
        Penjelasan Teknis:
        ------------------
        Operasi aras titik dengan contrast bekerja dengan mengalikan nilai piksel
        dengan suatu faktor skala. Nilai piksel gelap menjadi lebih gelap dan
        piksel terang menjadi lebih terang.
        
        Langkah-langkah:
        1. Kalikan setiap piksel dengan nilai alpha
        2. Clip nilai piksel ke rentang [0,255]
        3. Konversi ke tipe data uint8
        
        Parameters:
        -----------
        alpha : float
            Faktor pengali untuk kontras (> 1 meningkatkan kontras, < 1 menurunkan kontras)
            
        Returns:
        --------
        numpy.ndarray
            Gambar dengan kontras yang diubah
        """
        # Simpan langkah proses
        self.langkah_terakhir = f"""
        Proses Contrast Adjustment (Aras Titik):
        1. Setiap piksel dikalikan dengan faktor alpha {alpha}
        2. Nilai hasil dibatasi (clipping) antara 0-255
        3. Konversi ke tipe data uint8 standar untuk gambar
        """
        
        # Proses contrast adjustment
        hasil = np.clip(alpha * self.gambar_rgb, 0, 255).astype(np.uint8)
        return hasil
    
    def aras_titik_threshold(self, nilai=127):
        """
        Thresholding pada gambar grayscale (operasi titik)
        
        Penjelasan Teknis:
        ------------------
        Operasi aras titik dengan thresholding mengubah citra menjadi citra biner.
        Piksel dengan nilai di atas threshold menjadi putih (255) dan
        piksel dengan nilai di bawah threshold menjadi hitam (0).
        
        Langkah-langkah:
        1. Tentukan nilai threshold
        2. Bandingkan setiap piksel dengan nilai threshold
        3. Jika piksel > threshold, set nilai menjadi 255 (putih)
        4. Jika piksel <= threshold, set nilai menjadi 0 (hitam)
        
        Parameters:
        -----------
        nilai : int
            Nilai threshold (0-255)
            
        Returns:
        --------
        numpy.ndarray
            Gambar hasil thresholding
        """
        # Simpan langkah proses
        self.langkah_terakhir = f"""
        Proses Thresholding (Aras Titik):
        1. Bandingkan setiap piksel dengan nilai threshold {nilai}
        2. Jika nilai piksel > {nilai}, piksel menjadi putih (255)
        3. Jika nilai piksel <= {nilai}, piksel menjadi hitam (0)
        4. Hasilnya adalah citra biner hitam-putih
        """
        
        # Proses thresholding
        _, hasil = cv2.threshold(self.gambar_gray, nilai, 255, cv2.THRESH_BINARY)
        return hasil
    
    def aras_titik_negative(self):
        """
        Menghasilkan gambar negatif (operasi titik)
        
        Penjelasan Teknis:
        ------------------
        Operasi aras titik untuk menghasilkan negatif bekerja dengan
        membalikkan nilai setiap piksel dari rentang [0,255].
        Piksel gelap menjadi terang dan sebaliknya.
        
        Langkah-langkah:
        1. Kurangkan setiap nilai piksel dari 255
        
        Returns:
        --------
        numpy.ndarray
            Gambar negatif
        """
        # Simpan langkah proses
        self.langkah_terakhir = """
        Proses Negative Image (Aras Titik):
        1. Setiap piksel diinversi dengan rumus: piksel_baru = 255 - piksel_lama
        2. Piksel gelap menjadi terang, piksel terang menjadi gelap
        3. Warna diinversi: merah menjadi cyan, hijau menjadi magenta, biru menjadi kuning
        """
        
        # Proses inversi (negative)
        return 255 - self.gambar_rgb
    
    def aras_titik_histogram_equalization(self):
        """
        Ekualisasi histogram untuk meningkatkan kontras (operasi titik)
        
        Penjelasan Teknis:
        ------------------
        Ekualisasi histogram mendistribusikan ulang nilai piksel sehingga
        histogram intensitas menjadi seragam. Proses ini meningkatkan kontras
        dengan meratakan distribusi nilai-nilai keabuan.
        
        Langkah-langkah:
        1. Hitung histogram citra
        2. Normalisasi histogram untuk mendapatkan Cumulative Distribution Function (CDF)
        3. Gunakan CDF untuk memetakan nilai piksel ke rentang baru
        4. Terapkan pemetaan pada setiap piksel
        
        Returns:
        --------
        numpy.ndarray
            Gambar dengan histogram yang diekualisasi
        """
        # Simpan langkah proses
        self.langkah_terakhir = """
        Proses Histogram Equalization (Aras Titik):
        1. Hitung histogram (frekuensi kemunculan) nilai piksel untuk setiap channel warna
        2. Hitung Cumulative Distribution Function (CDF) dari histogram
        3. Normalisasi CDF untuk mendapatkan fungsi pemetaan
        4. Terapkan fungsi pemetaan ke setiap piksel
        5. Hasilnya adalah gambar dengan distribusi nilai piksel yang lebih merata dan kontras lebih baik
        """
        
        # Proses histogram equalization
        hasil = self.gambar_rgb.copy()
        for i in range(3):
            hasil[:,:,i] = cv2.equalizeHist(self.gambar_rgb[:,:,i])
        return hasil
    
    def demo_aras_titik(self):
        """
        Demo dan visualisasi semua operasi aras titik
        
        Returns:
        --------
        io.BytesIO
            Buffer yang berisi visualisasi gambar
        str
            Penjelasan proses
        """
        # Proses gambar
        gambar_brightness = self.aras_titik_brightness(50)
        gambar_contrast = self.aras_titik_contrast(1.5)
        gambar_negative = self.aras_titik_negative()
        gambar_threshold = self.aras_titik_threshold(127)
        gambar_histeq = self.aras_titik_histogram_equalization()
        
        # Penjelasan tahapan untuk setiap gambar
        tahapan = [
            "Gambar asli tanpa pemrosesan",
            "Nilai setiap piksel ditambah 50, sehingga gambar lebih cerah",
            "Nilai setiap piksel dikalikan 1.5, meningkatkan kontras",
            "Nilai setiap piksel diinversi: 255 - nilai_piksel",
            "Piksel > 127 menjadi putih (255), piksel <= 127 menjadi hitam (0)",
            "Distribusi nilai piksel diratakan untuk meningkatkan kontras"
        ]
        
        # Tampilkan gambar dengan penjelasan
        buf = self.tampilkan_gambar(
            [self.gambar_rgb, gambar_brightness, gambar_contrast, gambar_negative, gambar_threshold, gambar_histeq],
            ["Original", "Brightness +50", "Contrast 1.5x", "Negative", "Threshold", "Hist. Equalization"],
            (20, 6),
            tahapan
        )
        
        # Penjelasan teori aras titik
        penjelasan = """
        <h3>Aras Titik (Point Processing)</h3>
        <p>Pengolahan citra aras titik bekerja dengan memodifikasi nilai piksel secara individual tanpa mempertimbangkan
        piksel tetangga. Setiap piksel diproses secara independen dengan fungsi transformasi yang sama.</p>
        
        <h4>Karakteristik Aras Titik:</h4>
        <ul>
            <li>Bekerja pada piksel tunggal</li>
            <li>Tidak mempertimbangkan piksel tetangga</li>
            <li>Fungsi transformasi yang sama diterapkan pada semua piksel</li>
            <li>Operasi relatif cepat karena sederhana</li>
        </ul>
        
        <h4>Contoh Operasi Aras Titik:</h4>
        <ul>
            <li><strong>Brightness:</strong> Menambah atau mengurangi nilai piksel dengan konstanta</li>
            <li><strong>Contrast:</strong> Mengalikan nilai piksel dengan faktor skala</li>
            <li><strong>Thresholding:</strong> Mengubah gambar menjadi hitam-putih berdasarkan nilai ambang</li>
            <li><strong>Negative:</strong> Membalikkan nilai piksel untuk membuat gambar negatif</li>
            <li><strong>Histogram Equalization:</strong> Meratakan distribusi nilai piksel untuk meningkatkan kontras</li>
        </ul>
        
        <p>Rumus umum: g(x,y) = T[f(x,y)] dimana f adalah nilai piksel input, T adalah fungsi transformasi, dan g adalah nilai piksel output.</p>
        """
        
        return buf, penjelasan
    
    # ===== ARAS LOKAL =====
    def aras_lokal_gaussian_blur(self, sigma=3):
        """
        Gaussian blur untuk menghaluskan gambar (operasi lokal)
        
        Penjelasan Teknis:
        ------------------
        Gaussian blur adalah filter penghalusan yang menggunakan
        distribusi Gaussian sebagai kernel konvolusi. Filter ini
        mengambil rata-rata tertimbang dari piksel tetangga.
        
        Langkah-langkah:
        1. Buat kernel Gaussian 2D
        2. Lakukan konvolusi gambar dengan kernel
        
        Parameters:
        -----------
        sigma : float
            Parameter lebar distribusi Gaussian
            
        Returns:
        --------
        numpy.ndarray
            Gambar yang dihaluskan
        """
        # Simpan langkah proses
        self.langkah_terakhir = f"""
        Proses Gaussian Blur (Aras Lokal):
        1. Buat kernel konvolusi berdasarkan distribusi Gaussian 2D dengan sigma={sigma}
        2. Lakukan konvolusi antara gambar dan kernel (rata-rata tertimbang)
        3. Setiap piksel output adalah hasil rata-rata tertimbang piksel tetangga
        4. Piksel pada pusat kernel memiliki bobot lebih besar, semakin jauh bobot semakin kecil
        """
        
        # Proses gaussian blur
        return cv2.GaussianBlur(self.gambar_rgb, (0, 0), sigma)
    
    def aras_lokal_median_filter(self, ukuran=5):
        """
        Median filter untuk mengurangi noise (operasi lokal)
        
        Penjelasan Teknis:
        ------------------
        Median filter mengganti nilai piksel dengan nilai median
        dari piksel tetangga. Filter ini sangat efektif untuk
        mengurangi noise "salt and pepper".
        
        Langkah-langkah:
        1. Tentukan ukuran window (biasanya persegi, misalnya 3x3, 5x5)
        2. Untuk setiap piksel, ambil semua piksel dalam window
        3. Urutkan nilai-nilai tersebut dan ambil nilai tengahnya (median)
        4. Ganti nilai piksel dengan nilai median
        
        Parameters:
        -----------
        ukuran : int
            Ukuran window filter (ukuran x ukuran)
            
        Returns:
        --------
        numpy.ndarray
            Gambar yang difilter dengan median filter
        """
        # Simpan langkah proses
        self.langkah_terakhir = f"""
        Proses Median Filter (Aras Lokal):
        1. Untuk setiap piksel, ambil nilai piksel tetangga dalam window {ukuran}x{ukuran}
        2. Urutkan nilai-nilai tersebut dari kecil ke besar
        3. Ambil nilai tengah (median) dari hasil pengurutan
        4. Ganti nilai piksel dengan nilai median yang didapat
        5. Filter ini baik untuk menghilangkan noise "salt and pepper"
        """
        
        # Proses median filter
        return cv2.medianBlur(self.gambar_rgb, ukuran)
    
    def aras_lokal_sharpening(self, alpha=1.5):
        """
        Penajaman gambar dengan unsharp masking (operasi lokal)
        
        Penjelasan Teknis:
        ------------------
        Unsharp masking bekerja dengan mengurangi versi blur dari gambar
        dari gambar asli untuk mendapatkan "mask", lalu menambahkan
        mask tersebut ke gambar asli.
        
        Langkah-langkah:
        1. Blur gambar original (Gaussian blur)
        2. Hitung mask = original - blur
        3. Sharpened = original + alpha * mask
        
        Parameters:
        -----------
        alpha : float
            Faktor penajaman
            
        Returns:
        --------
        numpy.ndarray
            Gambar yang dipertajam
        """
        # Simpan langkah proses
        self.langkah_terakhir = f"""
        Proses Image Sharpening (Aras Lokal):
        1. Blur gambar asli dengan Gaussian blur untuk mendapatkan versi halus
        2. Hitung "mask" = gambar_asli - gambar_blur (berisi detail tajam)
        3. Kalikan mask dengan faktor alpha={alpha} untuk mengontrol tingkat penajaman
        4. Tambahkan mask yang sudah diskala ke gambar asli: gambar_tajam = gambar_asli + alpha*mask
        """
        
        # Proses sharpening
        blur = cv2.GaussianBlur(self.gambar_rgb, (0, 0), 3)
        return cv2.addWeighted(self.gambar_rgb, 1.0 + alpha, blur, -alpha, 0)
    
    def aras_lokal_gradient(self):
        """
        Deteksi tepi dengan gradient Sobel (operasi lokal)
        
        Penjelasan Teknis:
        ------------------
        Operator Sobel menghitung gradient citra dengan konvolusi
        menggunakan kernel khusus untuk arah x dan y. Magnitude
        gradient menunjukkan seberapa tajam perubahan intensitas.
        
        Langkah-langkah:
        1. Hitung gradient pada arah x menggunakan operator Sobel-x
        2. Hitung gradient pada arah y menggunakan operator Sobel-y
        3. Hitung magnitude gradient: sqrt(grad_x^2 + grad_y^2)
        
        Returns:
        --------
        numpy.ndarray
            Gambar hasil deteksi tepi
        """
        # Simpan langkah proses
        self.langkah_terakhir = """
        Proses Edge Detection dengan Gradient Sobel (Aras Lokal):
        1. Hitung gradient pada arah x dengan operator Sobel-x
           Kernel Sobel-x: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        2. Hitung gradient pada arah y dengan operator Sobel-y
           Kernel Sobel-y: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        3. Konversi ke nilai absolut (magnitude)
        4. Gabungkan gradient x dan y dengan rata-rata tertimbang
        5. Piksel dengan nilai tinggi menunjukkan tepi (perubahan intensitas yang tajam)
        """
        
        # Proses gradient detection
        grad_x = cv2.Sobel(self.gambar_gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(self.gambar_gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Konversi ke nilai absolut dan gabungkan
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        
        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        return grad
    
    def aras_lokal_adaptive_threshold(self, block_size=11, offset=2):
        """
        Adaptive threshold untuk segmentasi (operasi lokal)
        
        Penjelasan Teknis:
        ------------------
        Adaptive threshold menentukan nilai threshold berdasarkan
        statistik lokal pada area tetangga setiap piksel, sehingga
        adaptif terhadap variasi pencahayaan pada gambar.
        
        Langkah-langkah:
        1. Untuk setiap piksel, hitung rata-rata/gaussian weighted sum dalam area block_size
        2. Threshold = mean - offset
        3. Jika piksel > threshold, set nilai jadi 255, selainnya 0
        
        Parameters:
        -----------
        block_size : int
            Ukuran area tetangga untuk menghitung threshold
        offset : int
            Nilai yang dikurangkan dari rata-rata
            
        Returns:
        --------
        numpy.ndarray
            Gambar hasil adaptive threshold
        """
        # Simpan langkah proses
        self.langkah_terakhir = f"""
        Proses Adaptive Thresholding (Aras Lokal):
        1. Untuk setiap piksel, hitung mean/gaussian weighted average dari area {block_size}x{block_size}
        2. Tentukan threshold lokal = mean - {offset}
        3. Jika piksel > threshold lokal, set nilai jadi putih (255)
        4. Jika piksel <= threshold lokal, set nilai jadi hitam (0)
        5. Threshold adaptif menangani variasi pencahayaan lebih baik daripada threshold global
        """
        
        # Proses adaptive threshold
        return cv2.adaptiveThreshold(
            self.gambar_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, block_size, offset
        )
    
    def demo_aras_lokal(self):
        """
        Demo dan visualisasi semua operasi aras lokal
        
        Returns:
        --------
        io.BytesIO
            Buffer yang berisi visualisasi gambar
        str
            Penjelasan proses
        """
        # Proses gambar
        gambar_blur = self.aras_lokal_gaussian_blur(3)
        gambar_median = self.aras_lokal_median_filter(5)
        gambar_sharp = self.aras_lokal_sharpening(1.5)
        gambar_gradient = self.aras_lokal_gradient()
        gambar_adaptive = self.aras_lokal_adaptive_threshold()
        
        # Penjelasan tahapan untuk setiap gambar
        tahapan = [
            "Gambar asli tanpa pemrosesan",
            "Filter Gaussian dengan sigma=3 menghaluskan gambar dengan rata-rata tertimbang piksel tetangga",
            "Median filter dengan window 5x5 mengganti piksel dengan nilai median dari tetangga",
            "Unsharp masking menambah detail (gambar asli - versi blur) ke gambar asli untuk mempertajam",
            "Operator Sobel menghitung gradient (perubahan intensitas) untuk deteksi tepi",
            "Threshold adaptive menentukan nilai threshold berdasarkan statistik lokal area 11x11"
        ]
        
        # Tampilkan gambar dengan penjelasan
        buf = self.tampilkan_gambar(
            [self.gambar_rgb, gambar_blur, gambar_median, gambar_sharp, gambar_gradient, gambar_adaptive],
            ["Original", "Gaussian Blur", "Median Filter", "Sharpening", "Gradient (Sobel)", "Adaptive Threshold"],
            (20, 6),
            tahapan
        )
        
        # Penjelasan teori aras lokal
        penjelasan = """
        <h3>Aras Lokal (Local Processing)</h3>
        <p>Pengolahan citra aras lokal melibatkan analisis dan transformasi nilai piksel berdasarkan nilai
        piksel tetangga di sekitarnya. Biasanya menggunakan operasi konvolusi dengan berbagai kernel.</p>
        
        <h4>Karakteristik Aras Lokal:</h4>
        <ul>
            <li>Mempertimbangkan piksel tetangga dalam pemrosesan</li>
            <li>Menggunakan kernel atau mask konvolusi</li>
            <li>Dapat mendeteksi atau meningkatkan fitur spasial seperti tepi</li>
            <li>Lebih komputasional dibanding aras titik</li>
        </ul>
        
        <h4>Contoh Operasi Aras Lokal:</h4>
        <ul>
            <li><strong>Gaussian Blur:</strong> Menghaluskan gambar dengan filter Gaussian</li>
            <li><strong>Median Filter:</strong> Mengurangi noise dengan mengganti piksel dengan nilai median tetangga</li>
            <li><strong>Image Sharpening:</strong> Mempertajam detail dengan unsharp masking</li>
            <li><strong>Edge Detection:</strong> Mendeteksi tepi dengan operator gradient seperti Sobel</li>
            <li><strong>Adaptive Threshold:</strong> Threshold dengan nilai ambang yang berubah sesuai area lokal</li>
        </ul>
        
        <p>Konsep dasar: konvolusi antara citra input f(x,y) dengan kernel konvolusi h(x,y) menghasilkan citra output g(x,y).</p>
        """
        
        return buf, penjelasan
    
    # ===== ARAS GLOBAL =====
    def aras_global_fourier_transform(self):
        """
        Transformasi Fourier untuk melihat domain frekuensi (operasi global)
        
        Penjelasan Teknis:
        ------------------
        Transformasi Fourier mengubah representasi citra dari domain spasial
        ke domain frekuensi. Frekuensi tinggi merepresentasikan perubahan 
        cepat/detail halus, sedangkan frekuensi rendah merepresentasikan
        perubahan lambat/area halus.
        
        Langkah-langkah:
        1. Hitung 2D Discrete Fourier Transform (DFT)
        2. Geser hasil DFT sehingga frekuensi rendah berada di tengah
        3. Hitung magnitude spectrum dan konversi ke skala logaritmik
        
        Returns:
        --------
        numpy.ndarray
            Magnitude spectrum dari transformasi Fourier
        """
        # Simpan langkah proses
        self.langkah_terakhir = """
        Proses Transformasi Fourier (Aras Global):
        1. Hitung Discrete Fourier Transform 2D dari gambar grayscale
        2. Geser komponen frekuensi rendah ke tengah (fftshift)
        3. Hitung magnitude spectrum: |F(u,v)| = sqrt(Re(F)² + Im(F)²)
        4. Konversi ke skala logaritmik: 20*log(|F(u,v)|) untuk visualisasi lebih baik
        5. Titik tengah spektrum mewakili frekuensi rendah (komponen DC)
        6. Tepi spektrum mewakili frekuensi tinggi (detail halus, tepi)
        """
        
        # Proses Fourier Transform
        f_transform = np.fft.fft2(self.gambar_gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        magnitude_spectrum = np.asarray(magnitude_spectrum, dtype=np.uint8)
        return magnitude_spectrum
    
    def aras_global_low_pass_filter(self, radius=50):
        """
        Low-pass filter dalam domain frekuensi (operasi global)
        
        Penjelasan Teknis:
        ------------------
        Low-pass filter mempertahankan komponen frekuensi rendah dan
        membuang komponen frekuensi tinggi, menghasilkan efek blur.
        
        Langkah-langkah:
        1. Transformasi gambar ke domain frekuensi dengan DFT
        2. Buat mask lingkaran (radius menentukan cutoff frequency)
        3. Aplikasikan mask ke hasil DFT
        4. Transformasi balik ke domain spasial dengan Inverse DFT
        
        Parameters:
        -----------
        radius : int
            Radius mask (makin besar radius, makin sedikit blur)
            
        Returns:
        --------
        numpy.ndarray
            Gambar hasil filter low-pass
        """
        # Simpan langkah proses
        self.langkah_terakhir = f"""
        Proses Low-Pass Filter (Aras Global):
        1. Transformasi gambar ke domain frekuensi dengan DFT
        2. Geser komponen frekuensi rendah ke tengah (fftshift)
        3. Buat mask lingkaran dengan radius {radius} piksel (low-pass)
        4. Komponen dalam lingkaran (frekuensi rendah) dipertahankan
        5. Komponen luar lingkaran (frekuensi tinggi) dibuang
        6. Aplikasikan mask ke hasil DFT 
        7. Transformasi balik ke domain spasial dengan Inverse DFT
        8. Hasilnya adalah gambar yang lebih halus (blur)
        """
        
        # Proses Low Pass Filter
        rows, cols = self.gambar_gray.shape
        crow, ccol = rows // 2, cols // 2
        
        # Transformasi Fourier
        f_transform = np.fft.fft2(self.gambar_gray)
        f_shift = np.fft.fftshift(f_transform)
        
        # Buat mask
        mask = np.zeros((rows, cols), np.uint8)
        cv2.circle(mask, (ccol, crow), radius, 1, -1)
        
        # Filter dan inverse transform
        f_shift_filtered = f_shift * mask
        f_ishift = np.fft.ifftshift(f_shift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        
        # Normalisasi hasil
        img_result = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
        return img_result.astype(np.uint8)
    
    def aras_global_high_pass_filter(self, radius=50):
        """
        High-pass filter dalam domain frekuensi (operasi global)
        
        Penjelasan Teknis:
        ------------------
        High-pass filter mempertahankan komponen frekuensi tinggi dan
        membuang komponen frekuensi rendah, berguna untuk deteksi tepi
        dan penajaman gambar.
        
        Langkah-langkah:
        1. Transformasi gambar ke domain frekuensi dengan DFT
        2. Buat mask lingkaran inversi (komplemen dari low-pass)
        3. Aplikasikan mask ke hasil DFT
        4. Transformasi balik ke domain spasial dengan Inverse DFT
        
        Parameters:
        -----------
        radius : int
            Radius mask (makin kecil radius, makin ekstrim efek high-pass)
            
        Returns:
        --------
        numpy.ndarray
            Gambar hasil filter high-pass
        """
        # Simpan langkah proses
        self.langkah_terakhir = f"""
        Proses High-Pass Filter (Aras Global):
        1. Transformasi gambar ke domain frekuensi dengan DFT
        2. Geser komponen frekuensi rendah ke tengah (fftshift)
        3. Buat mask lingkaran inversi dengan radius {radius} piksel (high-pass)
        4. Komponen dalam lingkaran (frekuensi rendah) dibuang
        5. Komponen luar lingkaran (frekuensi tinggi) dipertahankan
        6. Aplikasikan mask ke hasil DFT
        7. Transformasi balik ke domain spasial dengan Inverse DFT
        8. Hasilnya adalah gambar dengan detail tepi yang dipertahankan
        """
        
        # Proses High Pass Filter
        rows, cols = self.gambar_gray.shape
        crow, ccol = rows // 2, cols // 2
        
        # Transformasi Fourier
        f_transform = np.fft.fft2(self.gambar_gray)
        f_shift = np.fft.fftshift(f_transform)
        
        # Buat mask kebalikan dari low-pass
        mask = np.ones((rows, cols), np.uint8)
        cv2.circle(mask, (ccol, crow), radius, 0, -1)
        
        # Filter dan inverse transform
        f_shift_filtered = f_shift * mask
        f_ishift = np.fft.ifftshift(f_shift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        
        # Normalisasi hasil
        img_result = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
        return img_result.astype(np.uint8)
    
    def aras_global_histogram_matching(self, target_gambar_path):
        """
        Histogram matching/specification (operasi global)
        
        Penjelasan Teknis:
        ------------------
        Histogram matching mengubah distribusi nilai piksel sehingga
        histogram gambar sumber cocok dengan histogram gambar target.
        
        Langkah-langkah:
        1. Hitung histogram gambar sumber dan target
        2. Hitung Cumulative Distribution Function (CDF) untuk keduanya
        3. Buat mapping berdasarkan CDF
        4. Transformasi gambar sumber menggunakan mapping
        
        Parameters:
        -----------
        target_gambar_path : str
            Path ke gambar target
            
        Returns:
        --------
        numpy.ndarray
            Gambar dengan histogram yang dicocokkan
        """
        # Muat gambar target
        target = None
        if isinstance(target_gambar_path, str):
            target = cv2.imread(target_gambar_path, 0)
        else:
            # Asumsi target_gambar_path sudah berupa array gambar
            if len(target_gambar_path.shape) == 3:
                target = cv2.cvtColor(target_gambar_path, cv2.COLOR_BGR2GRAY)
            else:
                target = target_gambar_path
                
        if target is None:
            raise ValueError("Gambar target tidak valid")
        
        # Simpan langkah proses
        self.langkah_terakhir = """
        Proses Histogram Matching (Aras Global):
        1. Hitung histogram gambar sumber (distribusi nilai piksel)
        2. Hitung histogram gambar target (distribusi nilai piksel)
        3. Hitung CDF untuk histogram sumber dan target
        4. Buat lookup table dengan memetakan CDF sumber ke CDF target
        5. Ganti nilai piksel gambar sumber dengan nilai dari lookup table
        6. Hasilnya adalah gambar dengan histogram yang mirip dengan gambar target
        """
        
        # Hitung histogram sumber dan target
        src_hist, _ = np.histogram(self.gambar_gray.flatten(), 256, [0, 256])
        tgt_hist, _ = np.histogram(target.flatten(), 256, [0, 256])
        
        # Hitung CDF sumber dan target
        src_cdf = src_hist.cumsum()
        src_cdf_normalized = src_cdf * float(src_hist.max()) / src_cdf.max()
        
        tgt_cdf = tgt_hist.cumsum()
        tgt_cdf_normalized = tgt_cdf * float(tgt_hist.max()) / tgt_cdf.max()
        
        # Buat lookup table untuk mapping
        lookup_table = np.zeros(256)
        for i in range(256):
            j = 255
            while j >= 0 and tgt_cdf_normalized[j] > src_cdf_normalized[i]:
                j -= 1
            lookup_table[i] = j
        
        # Terapkan lookup table
        hasil = np.uint8(lookup_table[self.gambar_gray])
        return hasil
    
    def demo_aras_global(self, target_gambar=None):
        """
        Demo dan visualisasi semua operasi aras global
        
        Parameters:
        -----------
        target_gambar : str atau numpy.ndarray, optional
            Path atau array gambar target untuk histogram matching
            
        Returns:
        --------
        io.BytesIO
            Buffer yang berisi visualisasi gambar
        str
            Penjelasan proses
        """
        # Proses gambar
        gambar_fourier = self.aras_global_fourier_transform()
        gambar_lowpass = self.aras_global_low_pass_filter(50)
        gambar_highpass = self.aras_global_high_pass_filter(50)
        
        gambar_list = [self.gambar_gray, gambar_fourier, gambar_lowpass, gambar_highpass]
        judul_list = ["Original (Gray)", "Fourier Transform", "Low Pass Filter", "High Pass Filter"]
        
        # Penjelasan tahapan untuk setiap gambar
        tahapan = [
            "Gambar asli grayscale",
            "Spektrum frekuensi: piksel terang di tengah = frekuensi rendah, tepi = frekuensi tinggi",
            "Low-pass filter mempertahankan frekuensi rendah (area halus) dan membuang frekuensi tinggi (detail)",
            "High-pass filter mempertahankan frekuensi tinggi (tepi) dan membuang frekuensi rendah"
        ]
        
        # Tambahkan histogram matching jika gambar target disediakan
        if target_gambar is not None:
            try:
                gambar_matched = self.aras_global_histogram_matching(target_gambar)
                gambar_list.append(gambar_matched)
                judul_list.append("Histogram Matching")
                tahapan.append("Histogram gambar sumber diubah agar sesuai dengan histogram gambar target")
            except Exception as e:
                print(f"Error dalam histogram matching: {e}")
        
        # Tampilkan gambar dengan penjelasan
        buf = self.tampilkan_gambar(gambar_list, judul_list, (20, 6), tahapan)
        
        # Penjelasan teori aras global
        penjelasan = """
        <h3>Aras Global (Global Processing)</h3>
        <p>Pengolahan citra aras global melibatkan transformasi dan analisis seluruh gambar sekaligus. 
        Nilai piksel output tergantung pada keseluruhan karakteristik gambar input.</p>
        
        <h4>Karakteristik Aras Global:</h4>
        <ul>
            <li>Mempertimbangkan seluruh gambar dalam pemrosesan</li>
            <li>Sering melibatkan transformasi antara domain spasial dan domain frekuensi</li>
            <li>Dapat menangkap dan memodifikasi pola frekuensi global</li>
            <li>Komputasi lebih kompleks daripada aras titik dan lokal</li>
        </ul>
        
        <h4>Contoh Operasi Aras Global:</h4>
        <ul>
            <li><strong>Transformasi Fourier:</strong> Mengubah gambar dari domain spasial ke domain frekuensi</li>
            <li><strong>Low-Pass Filter:</strong> Mempertahankan frekuensi rendah, menghasilkan efek blur</li>
            <li><strong>High-Pass Filter:</strong> Mempertahankan frekuensi tinggi, menonjolkan tepi dan detail</li>
            <li><strong>Histogram Matching:</strong> Mengubah distribusi nilai piksel agar sesuai dengan gambar target</li>
        </ul>
        
        <p>Fourier Transform memungkinkan kita memisahkan komponen frekuensi dalam gambar, 
        dengan rumus F(u,v) = ∑∑f(x,y)e^(-j2π(ux/M+vy/N)) untuk semua x,y.</p>
        """
        
        return buf, penjelasan
    
    # ===== ARAS OBJEK =====
    def aras_objek_segmentasi_kmeans(self, n_clusters=3):
        """
        Segmentasi dengan K-means clustering (operasi objek)
        
        Penjelasan Teknis:
        ------------------
        K-means clustering mengelompokkan piksel berdasarkan
        kesamaan warna, menghasilkan segmentasi objek dalam gambar.
        
        Langkah-langkah:
        1. Reshape gambar menjadi array 1D dari vektor fitur (RGB)
        2. Terapkan K-means clustering untuk mengelompokkan piksel
        3. Ganti nilai piksel dengan nilai pusat cluster
        4. Reshape kembali ke bentuk gambar asli
        
        Parameters:
        -----------
        n_clusters : int
            Jumlah cluster (segmen) yang diinginkan
            
        Returns:
        --------
        numpy.ndarray
            Gambar tersegmentasi
        """
        # Simpan langkah proses
        self.langkah_terakhir = f"""
        Proses Segmentasi K-means (Aras Objek):
        1. Reshape gambar RGB menjadi array 1D dari vektor 3D (R,G,B) 
        2. Terapkan algoritma K-means dengan {n_clusters} cluster
        3. Algoritma K-means:
           a. Pilih {n_clusters} pusat cluster awal secara acak
           b. Tetapkan setiap piksel ke cluster terdekat
           c. Hitung ulang pusat cluster
           d. Ulangi b-c hingga konvergen
        4. Ganti nilai setiap piksel dengan nilai pusat clusternya
        5. Hasilnya adalah gambar dengan {n_clusters} warna berbeda
        """
        
        # Reshape gambar untuk K-means
        pixels = self.gambar_rgb.reshape((-1, 3)).astype(np.float32)
        
        # Lakukan clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(pixels)
        centers = np.uint8(kmeans.cluster_centers_)
        
        # Kembalikan ke bentuk gambar
        segmented_image = centers[labels.flatten()]
        segmented_image = segmented_image.reshape(self.gambar_rgb.shape)
        return segmented_image
    
    def aras_objek_connected_components(self):
        """
        Analisis connected components (operasi objek)
        
        Penjelasan Teknis:
        ------------------
        Connected components melabeli region yang terhubung
        dalam gambar biner, berguna untuk menghitung dan
        menganalisis objek dalam gambar.
        
        Langkah-langkah:
        1. Konversi ke gambar biner (thresholding)
        2. Identifikasi piksel yang terhubung dengan algoritma flood fill
        3. Labeli setiap connected component dengan ID unik
        4. Visualisasikan dengan warna berbeda untuk setiap label
        
        Returns:
        --------
        numpy.ndarray
            Gambar dengan connected components yang diwarnai
        """
        # Simpan langkah proses
        self.langkah_terakhir = """
        Proses Connected Components (Aras Objek):
        1. Konversi gambar ke biner dengan metode threshold otomatis (Otsu)
        2. Lakukan analisis connected components:
           a. Mulai dengan label=0 untuk semua piksel
           b. Pindai gambar dan temukan piksel yang belum berlabel
           c. Lakukan flood fill untuk melabeli semua piksel yang terhubung
           d. Setiap kumpulan piksel terhubung mendapat label unik
        3. Hitung statistik untuk setiap connected component (area, posisi, dll)
        4. Visualisasikan setiap component dengan warna berbeda
        """
        
        # Konversi ke binary image
        _, binary = cv2.threshold(self.gambar_gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        
        # Analisis connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        # Buat gambar output dengan warna
        output = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
        
        # Warnai tiap komponen dengan warna berbeda
        np.random.seed(42)  # Untuk konsistensi warna
        for i in range(1, num_labels):  # Skip label 0 (background)
            mask = labels == i
            # Buat warna random untuk tiap komponen
            color = np.random.randint(0, 255, size=3)
            output[mask] = color
        
        return output
    
    def aras_objek_watershed(self):
        """
        Segmentasi dengan algoritma watershed (operasi objek)
        
        Penjelasan Teknis:
        ------------------
        Watershed menggunakan konsep topografi untuk segmentasi.
        Gambar grayscale diperlakukan sebagai permukaan topografi,
        dan 'air' mengisi basin dari sumber (marker).
        
        Langkah-langkah:
        1. Hapus noise dengan morphological opening
        2. Temukan background pasti dengan dilasi
        3. Temukan foreground pasti dengan distance transform dan threshold
        4. Temukan unknown region (bkgd - frgd)
        5. Lakukan labelling dengan connected components
        6. Aplikasikan watershed
        
        Returns:
        --------
        numpy.ndarray
            Gambar hasil segmentasi watershed
        """
        # Simpan langkah proses
        self.langkah_terakhir = """
        Proses Watershed Segmentation (Aras Objek):
        1. Penghilangan noise dengan morphological opening
        2. Identifikasi background pasti dengan operasi dilasi
        3. Identifikasi foreground pasti dengan distance transform:
           a. Hitung jarak setiap piksel foreground ke background terdekat
           b. Threshold hasil untuk mendapatkan marker foreground
        4. Identifikasi unknown region = background - foreground
        5. Lakukan connected components labelling untuk marker
        6. Terapkan algoritma watershed:
           a. Pertimbangkan gambar sebagai topografi
           b. Mulai 'aliran air' dari marker
           c. 'Air' mengalir dan mengisi 'cekungan' (objek)
           d. 'Barrier' dibuat saat air dari marker berbeda bertemu
        7. Hasilnya adalah gambar dengan batas objek yang ditandai
        """
        
        # Noise removal
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(self.gambar_gray, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Tentukan background
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Tentukan foreground dengan distance transform
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
        
        # Unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)
        
        # Tambahkan 1 ke semua label sehingga background bukan 0
        markers = markers + 1
        
        # Unknown region sekarang bertanda 0
        markers[unknown == 255] = 0
        
        # Lakukan watershed
        markers = cv2.watershed(self.gambar_original, markers)
        
        # Visualisasi hasil
        hasil = self.gambar_rgb.copy()
        hasil[markers == -1] = [255, 0, 0]  # Batas dengan warna merah
        
        return hasil
    
    def aras_objek_contour_detection(self):
        """
        Deteksi kontur objek (operasi objek)
        
        Penjelasan Teknis:
        ------------------
        Kontur adalah kurva yang menghubungkan semua titik kontinu
        sepanjang batas dengan warna atau intensitas yang sama.
        Berguna untuk analisis bentuk dan deteksi objek.
        
        Langkah-langkah:
        1. Konversi gambar ke biner (thresholding)
        2. Temukan kontur dari gambar biner
        3. Gambar kontur pada gambar asli
        
        Returns:
        --------
        numpy.ndarray
            Gambar dengan kontur yang terdeteksi
        """
        # Simpan langkah proses
        self.langkah_terakhir = """
        Proses Contour Detection (Aras Objek):
        1. Konversi gambar ke biner dengan metode threshold otomatis (Otsu)
        2. Cari kontur pada gambar biner:
           a. Kontur adalah kurva yang menghubungkan piksel dengan nilai sama pada batas
           b. Algoritma melakukan tracing pada batas antara piksel putih dan hitam
        3. Gambar kontur yang ditemukan pada citra asli dengan warna hijau
        4. Kontur merepresentasikan batas objek dalam gambar
        """
        
        # Konversi ke binary image
        _, binary = cv2.threshold(self.gambar_gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        
        # Temukan kontur
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Gambar kontur
        hasil = self.gambar_rgb.copy()
        cv2.drawContours(hasil, contours, -1, (0, 255, 0), 2)
        
        return hasil
    
    def demo_aras_objek(self):
        """
        Demo dan visualisasi semua operasi aras objek
        
        Returns:
        --------
        io.BytesIO
            Buffer yang berisi visualisasi gambar
        str
            Penjelasan proses
        """
        # Proses gambar
        gambar_kmeans = self.aras_objek_segmentasi_kmeans(3)
        gambar_components = self.aras_objek_connected_components()
        gambar_watershed = self.aras_objek_watershed()
        gambar_contours = self.aras_objek_contour_detection()
        
        # Penjelasan tahapan untuk setiap gambar
        tahapan = [
            "Gambar asli tanpa pemrosesan",
            "K-means clustering mengelompokkan piksel dengan warna serupa menjadi 3 cluster",
            "Connected components melabeli region terhubung dengan warna berbeda",
            "Watershed menggunakan konsep topografi untuk segmentasi objek (batas merah)",
            "Contour detection mengidentifikasi dan menggambar batas objek (hijau)"
        ]
        
        # Tampilkan gambar dengan penjelasan
        buf = self.tampilkan_gambar(
            [self.gambar_rgb, gambar_kmeans, gambar_components, gambar_watershed, gambar_contours],
            ["Original", "K-Means Segmentation", "Connected Components", "Watershed Segmentation", "Contour Detection"],
            (20, 6),
            tahapan
        )
        
        # Penjelasan teori aras objek
        penjelasan = """
        <h3>Aras Objek (Object Processing)</h3>
        <p>Pengolahan citra aras objek fokus pada analisis dan manipulasi kumpulan piksel
        yang membentuk objek atau region tertentu dalam gambar. Pendekatan ini memperhatikan
        semantik dan struktur objek, bukan hanya nilai piksel.</p>
        
        <h4>Karakteristik Aras Objek:</h4>
        <ul>
            <li>Berfokus pada region atau objek yang bermakna</li>
            <li>Melibatkan segmentasi dan ekstraksi fitur</li>
            <li>Mempertimbangkan hubungan spasial antar piksel</li>
            <li>Level abstraksi yang lebih tinggi daripada aras titik, lokal, dan global</li>
        </ul>
        
        <h4>Contoh Operasi Aras Objek:</h4>
        <ul>
            <li><strong>K-means Segmentation:</strong> Mengelompokkan piksel dengan karakteristik serupa</li>
            <li><strong>Connected Components:</strong> Melabeli region terhubung dalam gambar biner</li>
            <li><strong>Watershed Segmentation:</strong> Menggunakan konsep topografi untuk segmentasi</li>
            <li><strong>Contour Detection:</strong> Mengidentifikasi batas antara objek dan background</li>
        </ul>
        
        <p>Pada aras objek, pemrosesan bergeser dari piksel individual ke region bermakna yang 
        merepresentasikan entitas dunia nyata dalam gambar.</p>
        """
        
        return buf, penjelasan
    
    def demo_semua_aras(self, target_gambar=None):
        """
        Tampilkan demo semua jenis aras
        
        Parameters:
        -----------
        target_gambar : str atau numpy.ndarray, optional
            Path atau array gambar target untuk histogram matching
            
        Returns:
        --------
        list
            Daftar buffer gambar dan penjelasan
        """
        hasil = []
        
        print("Demo Teknik Pengolahan Citra Digital")
        print("=" * 40)
        
        print("\n1. Aras Titik (Point Processing)")
        buf, penjelasan = self.demo_aras_titik()
        hasil.append((buf, "Aras Titik", penjelasan))
        
        print("\n2. Aras Lokal (Local Processing)")
        buf, penjelasan = self.demo_aras_lokal()
        hasil.append((buf, "Aras Lokal", penjelasan))
        
        print("\n3. Aras Global (Global Processing)")
        buf, penjelasan = self.demo_aras_global(target_gambar)
        hasil.append((buf, "Aras Global", penjelasan))
        
        print("\n4. Aras Objek (Object Processing)")
        buf, penjelasan = self.demo_aras_objek()
        hasil.append((buf, "Aras Objek", penjelasan))
        
        print("\nDemo selesai!")
        return hasil


# === APLIKASI WEB DENGAN FLASK ===
app = Flask(__name__)
pengolah = None

@app.route('/')
def index():
    """Tampilkan halaman utama"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle upload gambar"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Tidak ada file yang diunggah'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Tidak ada file yang dipilih'}), 400
            
        if file:
            # Simpan file sementara
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            file.save(temp_file.name)
            
            # Proses gambar
            global pengolah
            pengolah = PembelajaranPengolahanCitra(temp_file.name)
            
            # Konversi gambar original ke base64 untuk ditampilkan
            gambar_buf = pengolah.tampilkan_gambar([pengolah.gambar_rgb], ["Gambar Original"])
            gambar_base64 = pengolah.gambar_ke_base64(gambar_buf)
            
            return jsonify({
                'success': True,
                'message': 'Gambar berhasil diunggah',
                'gambar': gambar_base64,
                'ukuran': pengolah.gambar_original.shape[:2]
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/proses/<jenis_aras>/<teknik>', methods=['POST'])
def proses_gambar(jenis_aras, teknik):
    """Proses gambar dengan teknik tertentu"""
    try:
        if pengolah is None:
            return jsonify({'error': 'Tidak ada gambar yang diunggah'}), 400
            
        # Parameter dari request
        params = request.json or {}
        
        # Proses sesuai jenis aras dan teknik
        if jenis_aras == 'titik':
            if teknik == 'brightness':
                nilai = int(params.get('nilai', 50))
                hasil = pengolah.aras_titik_brightness(nilai)
                judul = f"Brightness Adjustment ({nilai})"
            elif teknik == 'contrast':
                alpha = float(params.get('alpha', 1.5))
                hasil = pengolah.aras_titik_contrast(alpha)
                judul = f"Contrast Adjustment ({alpha}x)"
            elif teknik == 'threshold':
                nilai = int(params.get('nilai', 127))
                hasil = pengolah.aras_titik_threshold(nilai)
                judul = f"Threshold (nilai={nilai})"
            elif teknik == 'negative':
                hasil = pengolah.aras_titik_negative()
                judul = "Negative Image"
            elif teknik == 'hist_eq':
                hasil = pengolah.aras_titik_histogram_equalization()
                judul = "Histogram Equalization"
            else:
                return jsonify({'error': 'Teknik tidak dikenal'}), 400
                
        elif jenis_aras == 'lokal':
            if teknik == 'gaussian_blur':
                sigma = float(params.get('sigma', 3.0))
                hasil = pengolah.aras_lokal_gaussian_blur(sigma)
                judul = f"Gaussian Blur (sigma={sigma})"
            elif teknik == 'median_filter':
                ukuran = int(params.get('ukuran', 5))
                hasil = pengolah.aras_lokal_median_filter(ukuran)
                judul = f"Median Filter ({ukuran}x{ukuran})"
            elif teknik == 'sharpening':
                alpha = float(params.get('alpha', 1.5))
                hasil = pengolah.aras_lokal_sharpening(alpha)
                judul = f"Sharpening (alpha={alpha})"
            elif teknik == 'gradient':
                hasil = pengolah.aras_lokal_gradient()
                judul = "Gradient (Sobel)"
            elif teknik == 'adaptive_threshold':
                block_size = int(params.get('block_size', 11))
                offset = int(params.get('offset', 2))
                hasil = pengolah.aras_lokal_adaptive_threshold(block_size, offset)
                judul = f"Adaptive Threshold (block={block_size}, offset={offset})"
            else:
                return jsonify({'error': 'Teknik tidak dikenal'}), 400
                
        elif jenis_aras == 'global':
            if teknik == 'fourier':
                hasil = pengolah.aras_global_fourier_transform()
                judul = "Fourier Transform"
            elif teknik == 'lowpass':
                radius = int(params.get('radius', 50))
                hasil = pengolah.aras_global_low_pass_filter(radius)
                judul = f"Low Pass Filter (radius={radius})"
            elif teknik == 'highpass':
                radius = int(params.get('radius', 50))
                hasil = pengolah.aras_global_high_pass_filter(radius)
                judul = f"High Pass Filter (radius={radius})"
            else:
                return jsonify({'error': 'Teknik tidak dikenal'}), 400
                
        elif jenis_aras == 'objek':
            if teknik == 'kmeans':
                n_clusters = int(params.get('n_clusters', 3))
                hasil = pengolah.aras_objek_segmentasi_kmeans(n_clusters)
                judul = f"K-Means Segmentation (k={n_clusters})"
            elif teknik == 'components':
                hasil = pengolah.aras_objek_connected_components()
                judul = "Connected Components"
            elif teknik == 'watershed':
                hasil = pengolah.aras_objek_watershed()
                judul = "Watershed Segmentation"
            elif teknik == 'contour':
                hasil = pengolah.aras_objek_contour_detection()
                judul = "Contour Detection"
            else:
                return jsonify({'error': 'Teknik tidak dikenal'}), 400
        else:
            return jsonify({'error': 'Jenis aras tidak dikenal'}), 400
            
        # Tampilkan hasil
        gambar_buf = pengolah.tampilkan_gambar(
            [pengolah.gambar_rgb, hasil], 
            ["Original", judul]
        )
        gambar_base64 = pengolah.gambar_ke_base64(gambar_buf)
        
        return jsonify({
            'success': True,
            'gambar': gambar_base64,
            'penjelasan': pengolah.langkah_terakhir
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/demo/<jenis_aras>', methods=['GET'])
def demo_aras(jenis_aras):
    """Tampilkan demo untuk jenis aras tertentu"""
    try:
        if pengolah is None:
            return jsonify({'error': 'Tidak ada gambar yang diunggah'}), 400
            
        if jenis_aras == 'titik':
            buf, penjelasan = pengolah.demo_aras_titik()
        elif jenis_aras == 'lokal':
            buf, penjelasan = pengolah.demo_aras_lokal()
        elif jenis_aras == 'global':
            buf, penjelasan = pengolah.demo_aras_global()
        elif jenis_aras == 'objek':
            buf, penjelasan = pengolah.demo_aras_objek()
        else:
            return jsonify({'error': 'Jenis aras tidak dikenal'}), 400
            
        gambar_base64 = pengolah.gambar_ke_base64(buf)
        
        return jsonify({
            'success': True,
            'gambar': gambar_base64,
            'penjelasan': penjelasan
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/demo_semua', methods=['GET'])
def demo_semua_aras():
    """Tampilkan demo untuk semua jenis aras"""
    try:
        if pengolah is None:
            return jsonify({'error': 'Tidak ada gambar yang diunggah'}), 400
            
        hasil_demo = pengolah.demo_semua_aras()
        
        # Konversi semua buffer gambar ke base64
        hasil_base64 = []
        for buf, judul, penjelasan in hasil_demo:
            hasil_base64.append({
                'gambar': pengolah.gambar_ke_base64(buf),
                'judul': judul,
                'penjelasan': penjelasan
            })
        
        return jsonify({
            'success': True,
            'hasil': hasil_base64
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/preview/<jenis_aras>/<teknik>', methods=['POST'])
def preview_gambar(jenis_aras, teknik):
    """Handle preview gambar dengan parameter"""
    try:
        if pengolah is None:
            return jsonify({'error': 'Tidak ada gambar yang diunggah'}), 400
            
        params = request.json or {}
        hasil = proses_preview(jenis_aras, teknik, params)  # Fungsi khusus preview
        gambar_base64 = pengolah.gambar_ke_base64(hasil)
        
        return jsonify({
            'success': True,
            'gambar': gambar_base64
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/histogram_matching', methods=['POST'])
def histogram_matching():
    """Proses histogram matching dengan gambar target"""
    try:
        if pengolah is None:
            return jsonify({'error': 'Tidak ada gambar sumber yang diunggah'}), 400
            
        if 'file' not in request.files:
            return jsonify({'error': 'Tidak ada file target yang diunggah'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Tidak ada file target yang dipilih'}), 400
            
        # Simpan file target sementara
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        file.save(temp_file.name)
        
        # Proses histogram matching
        target_gambar = cv2.imread(temp_file.name)
        if target_gambar is None:
            return jsonify({'error': 'Gambar target tidak valid'}), 400
            
        hasil = pengolah.aras_global_histogram_matching(target_gambar)
        
        # Tampilkan hasil
        target_gray = cv2.cvtColor(target_gambar, cv2.COLOR_BGR2GRAY)
        target_rgb = cv2.cvtColor(target_gambar, cv2.COLOR_BGR2RGB)
        
        gambar_buf = pengolah.tampilkan_gambar(
            [pengolah.gambar_gray, target_gray, hasil], 
            ["Gambar Sumber", "Gambar Target", "Hasil Histogram Matching"]
        )
        gambar_base64 = pengolah.gambar_ke_base64(gambar_buf)
        
        return jsonify({
            'success': True,
            'gambar': gambar_base64,
            'penjelasan': pengolah.langkah_terakhir
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Template HTML untuk frontend
@app.route('/templates/index.html')
def get_index_template():
    """Render template HTML untuk halaman utama"""
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
