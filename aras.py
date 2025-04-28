import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature, measure, segmentation
from sklearn.cluster import KMeans
from scipy import ndimage

class PengolahanCitra:
    def __init__(self, gambar_path):
        """
        Inisialisasi kelas pengolahan citra dengan path gambar
        """
        self.gambar_original = cv2.imread(gambar_path)
        if self.gambar_original is None:
            raise FileNotFoundError(f"Gambar tidak ditemukan: {gambar_path}")
        
        # Konversi BGR ke RGB untuk matplotlib
        self.gambar_rgb = cv2.cvtColor(self.gambar_original, cv2.COLOR_BGR2RGB)
        
        # Konversi ke grayscale
        self.gambar_gray = cv2.cvtColor(self.gambar_original, cv2.COLOR_BGR2GRAY)
        
        print(f"Gambar berhasil dimuat dengan ukuran: {self.gambar_original.shape}")
    
    def tampilkan_gambar(self, gambar_list, judul_list, ukuran=(15, 10)):
        """
        Menampilkan beberapa gambar dengan pyplot
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
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    # ===== ARAS TITIK =====
    def aras_titik_brightness(self, nilai=50):
        """
        Mengubah kecerahan gambar (operasi titik)
        """
        hasil = np.clip(self.gambar_rgb.astype(np.int16) + nilai, 0, 255).astype(np.uint8)
        return hasil
    
    def aras_titik_contrast(self, alpha=1.5):
        """
        Mengubah kontras gambar (operasi titik)
        """
        hasil = np.clip(alpha * self.gambar_rgb, 0, 255).astype(np.uint8)
        return hasil
    
    def aras_titik_threshold(self, nilai=127):
        """
        Thresholding (operasi titik)
        """
        _, hasil = cv2.threshold(self.gambar_gray, nilai, 255, cv2.THRESH_BINARY)
        return hasil
    
    def aras_titik_negative(self):
        """
        Menghasilkan gambar negatif (operasi titik)
        """
        return 255 - self.gambar_rgb
    
    def aras_titik_histogram_equalization(self):
        """
        Ekualisasi histogram (operasi titik)
        """
        hasil = self.gambar_rgb.copy()
        for i in range(3):
            hasil[:,:,i] = cv2.equalizeHist(self.gambar_rgb[:,:,i])
        return hasil
    
    def demo_aras_titik(self):
        """
        Demo semua operasi aras titik
        """
        gambar_brightness = self.aras_titik_brightness(50)
        gambar_contrast = self.aras_titik_contrast(1.5)
        gambar_negative = self.aras_titik_negative()
        gambar_threshold = self.aras_titik_threshold(127)
        gambar_histeq = self.aras_titik_histogram_equalization()
        
        self.tampilkan_gambar(
            [self.gambar_rgb, gambar_brightness, gambar_contrast, gambar_negative, gambar_threshold, gambar_histeq],
            ["Original", "Brightness +50", "Contrast 1.5x", "Negative", "Threshold", "Hist. Equalization"],
            (20, 10)
        )
    
    # ===== ARAS LOKAL =====
    def aras_lokal_gaussian_blur(self, sigma=3):
        """
        Gaussian blur (operasi lokal)
        """
        return cv2.GaussianBlur(self.gambar_rgb, (0, 0), sigma)
    
    def aras_lokal_median_filter(self, ukuran=5):
        """
        Median filter (operasi lokal)
        """
        return cv2.medianBlur(self.gambar_rgb, ukuran)
    
    def aras_lokal_sharpening(self, alpha=1.5):
        """
        Penajaman gambar dengan unsharp masking (operasi lokal)
        """
        blur = cv2.GaussianBlur(self.gambar_rgb, (0, 0), 3)
        return cv2.addWeighted(self.gambar_rgb, 1.0 + alpha, blur, -alpha, 0)
    
    def aras_lokal_gradient(self):
        """
        Deteksi tepi dengan gradient Sobel (operasi lokal)
        """
        grad_x = cv2.Sobel(self.gambar_gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(self.gambar_gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Konversi ke nilai absolut dan gabungkan
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        
        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        return grad
    
    def aras_lokal_adaptive_threshold(self, block_size=11, offset=2):
        """
        Adaptive threshold (operasi lokal)
        """
        return cv2.adaptiveThreshold(
            self.gambar_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, block_size, offset
        )
    
    def demo_aras_lokal(self):
        """
        Demo semua operasi aras lokal
        """
        gambar_blur = self.aras_lokal_gaussian_blur(3)
        gambar_median = self.aras_lokal_median_filter(5)
        gambar_sharp = self.aras_lokal_sharpening(1.5)
        gambar_gradient = self.aras_lokal_gradient()
        gambar_adaptive = self.aras_lokal_adaptive_threshold()
        
        self.tampilkan_gambar(
            [self.gambar_rgb, gambar_blur, gambar_median, gambar_sharp, gambar_gradient, gambar_adaptive],
            ["Original", "Gaussian Blur", "Median Filter", "Sharpening", "Gradient (Sobel)", "Adaptive Threshold"],
            (20, 10)
        )
    
    # ===== ARAS GLOBAL =====
    def aras_global_fourier_transform(self):
        """
        Transformasi Fourier (operasi global)
        """
        f_transform = np.fft.fft2(self.gambar_gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        magnitude_spectrum = np.asarray(magnitude_spectrum, dtype=np.uint8)
        return magnitude_spectrum
    
    def aras_global_low_pass_filter(self, radius=50):
        """
        Low-pass filter dalam domain frekuensi (operasi global)
        """
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
        """
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
        """
        target = cv2.imread(target_gambar_path, 0)
        if target is None:
            raise FileNotFoundError(f"Gambar target tidak ditemukan: {target_gambar_path}")
        
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
    
    def demo_aras_global(self, target_gambar_path=None):
        """
        Demo semua operasi aras global
        """
        gambar_fourier = self.aras_global_fourier_transform()
        gambar_lowpass = self.aras_global_low_pass_filter(50)
        gambar_highpass = self.aras_global_high_pass_filter(50)
        
        gambar_list = [self.gambar_gray, gambar_fourier, gambar_lowpass, gambar_highpass]
        judul_list = ["Original (Gray)", "Fourier Transform", "Low Pass Filter", "High Pass Filter"]
        
        if target_gambar_path:
            try:
                gambar_matched = self.aras_global_histogram_matching(target_gambar_path)
                gambar_list.append(gambar_matched)
                judul_list.append("Histogram Matching")
            except Exception as e:
                print(f"Error dalam histogram matching: {e}")
        
        self.tampilkan_gambar(gambar_list, judul_list, (20, 10))
    
    # ===== ARAS OBJEK =====
    def aras_objek_segmentasi_kmeans(self, n_clusters=3):
        """
        Segmentasi dengan K-means clustering (operasi objek)
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
        """
        # Konversi ke binary image
        _, binary = cv2.threshold(self.gambar_gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        
        # Analisis connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        # Buat gambar output dengan warna
        output = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
        
        # Warnai tiap komponen dengan warna berbeda
        for i in range(1, num_labels):  # Skip label 0 (background)
            mask = labels == i
            # Buat warna random untuk tiap komponen
            color = np.random.randint(0, 255, size=3)
            output[mask] = color
        
        return output
    
    def aras_objek_watershed(self):
        """
        Segmentasi dengan algoritma watershed (operasi objek)
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
        Demo semua operasi aras objek
        """
        gambar_kmeans = self.aras_objek_segmentasi_kmeans(3)
        gambar_components = self.aras_objek_connected_components()
        gambar_watershed = self.aras_objek_watershed()
        gambar_contours = self.aras_objek_contour_detection()
        
        self.tampilkan_gambar(
            [self.gambar_rgb, gambar_kmeans, gambar_components, gambar_watershed, gambar_contours],
            ["Original", "K-Means Segmentation", "Connected Components", "Watershed Segmentation", "Contour Detection"],
            (20, 10)
        )
    
    def demo_semua_aras(self, target_gambar_path=None):
        """
        Tampilkan demo semua jenis aras
        """
        print("Demo Teknik Pengolahan Citra Digital")
        print("=" * 40)
        
        print("\n1. Aras Titik (Point Processing)")
        self.demo_aras_titik()
        
        print("\n2. Aras Lokal (Local Processing)")
        self.demo_aras_lokal()
        
        print("\n3. Aras Global (Global Processing)")
        self.demo_aras_global(target_gambar_path)
        
        print("\n4. Aras Objek (Object Processing)")
        self.demo_aras_objek()
        
        print("\nDemo selesai!")


# Contoh penggunaan
if __name__ == "__main__":
    # Ganti dengan path gambar yang ingin diproses
    gambar_path = "gumbalask.png"
    target_path = "gambar_target.jpg"  # Untuk histogram matching
    
    try:
        pengolah = PengolahanCitra(gambar_path)
        
        # Demo semua aras secara berurutan
        pengolah.demo_semua_aras(target_path)
        
        # Atau bisa juga demo individual:
        # pengolah.demo_aras_titik()
        # pengolah.demo_aras_lokal()
        # pengolah.demo_aras_global()
        # pengolah.demo_aras_objek()
        
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")
        print("Pastikan file gambar tersedia dan path sudah benar.")