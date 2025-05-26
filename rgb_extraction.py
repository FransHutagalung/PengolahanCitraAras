import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

class RGBImageAnalyzer:
    def __init__(self):
        self.image = None
        self.image_rgb = None
        self.filename = None
        
    def load_image(self, filepath=None):
        """Load image dari file path atau dialog"""
        if filepath is None:
            root = tk.Tk()
            root.withdraw()
            filepath = filedialog.askopenfilename(
                title="Pilih Gambar",
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
            )
            root.destroy()
        
        if filepath:
            self.filename = filepath
            self.image = cv2.imread(filepath)
            self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            print(f"✅ Gambar berhasil dimuat: {filepath}")
            print(f"📐 Ukuran gambar: {self.image_rgb.shape}")
            return True
        return False
    
    def get_pixel_rgb(self, x, y):
        """Ambil nilai RGB pada koordinat tertentu"""
        if self.image_rgb is not None:
            if 0 <= y < self.image_rgb.shape[0] and 0 <= x < self.image_rgb.shape[1]:
                r, g, b = self.image_rgb[y, x]
                return int(r), int(g), int(b)
        return None
    
    def get_region_rgb(self, x1, y1, x2, y2):
        """Ambil nilai RGB dari region/area tertentu"""
        if self.image_rgb is not None:
            region = self.image_rgb[y1:y2, x1:x2]
            return region
        return None
    
    def analyze_color_channels(self):
        """Analisis statistik setiap channel warna"""
        if self.image_rgb is None:
            print("❌ Belum ada gambar yang dimuat!")
            return None
        
        r_channel = self.image_rgb[:, :, 0]
        g_channel = self.image_rgb[:, :, 1]
        b_channel = self.image_rgb[:, :, 2]
        
        stats = {
            'Red': {
                'mean': np.mean(r_channel),
                'std': np.std(r_channel),
                'min': np.min(r_channel),
                'max': np.max(r_channel),
                'median': np.median(r_channel)
            },
            'Green': {
                'mean': np.mean(g_channel),
                'std': np.std(g_channel),
                'min': np.min(g_channel),
                'max': np.max(g_channel),
                'median': np.median(g_channel)
            },
            'Blue': {
                'mean': np.mean(b_channel),
                'std': np.std(b_channel),
                'min': np.min(b_channel),
                'max': np.max(b_channel),
                'median': np.median(b_channel)
            }
        }
        
        return stats
    
    # def create_histogram(self):
    #     """Membuat histogram untuk setiap channel RGB"""
    #     if self.image_rgb is None:
    #         return None
        
    #     fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
    #     colors = ['red', 'green', 'blue']
    #     for i, color in enumerate(colors):
    #         axes[0, i].hist(self.image_rgb[:, :, i].ravel(), bins=256, 
    #                        color=color, alpha=0.7)
    #         axes[0, i].set_title(f'Histogram Channel {color.upper()}')
    #         axes[0, i].set_xlabel('Intensitas Pixel')
    #         axes[0, i].set_ylabel('Frekuensi')
        
    #     for i, color in enumerate(colors):
    #         axes[1, 0].hist(self.image_rgb[:, :, i].ravel(), bins=256, 
    #                        color=color, alpha=0.5, label=color.upper())
    #     axes[1, 0].set_title('Histogram Gabungan RGB')
    #     axes[1, 0].set_xlabel('Intensitas Pixel')
    #     axes[1, 0].set_ylabel('Frekuensi')
    #     axes[1, 0].legend()
        
    #     axes[1, 1].imshow(self.image_rgb)
    #     axes[1, 1].set_title('Gambar Asli')
    #     axes[1, 1].axis('off')
        
    #     plt.tight_layout()
    #     return fig
    
    def create_histogram(self):
        """Membuat histogram untuk setiap channel RGB"""
        if self.image_rgb is None:
            return None
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        colors = ['red', 'green', 'blue']
        for i, color in enumerate(colors):
            axes[0, i].hist(self.image_rgb[:, :, i].ravel(), bins=256, 
                           color=color, alpha=0.7)
            axes[0, i].set_title(f'Histogram Channel {color.upper()}')
            axes[0, i].set_xlabel('Intensitas Pixel')
            axes[0, i].set_ylabel('Frekuensi')
            axes[0, i].grid(True, alpha=0.3)
        
        for i, color in enumerate(colors):
            axes[1, 0].hist(self.image_rgb[:, :, i].ravel(), bins=256, 
                           color=color, alpha=0.5, label=color.upper())
        axes[1, 0].set_title('Histogram Gabungan RGB')
        axes[1, 0].set_xlabel('Intensitas Pixel')
        axes[1, 0].set_ylabel('Frekuensi')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].imshow(self.image_rgb)
        axes[1, 1].set_title('Gambar Asli')
        axes[1, 1].axis('off')
        
        # Statistik RGB
        stats = self.analyze_color_channels()
        stats_text = ""
        for channel, data in stats.items():
            stats_text += f"{channel}:\n"
            stats_text += f"  Mean: {data['mean']:.1f}\n"
            stats_text += f"  Std: {data['std']:.1f}\n"
            stats_text += f"  Range: {data['min']}-{data['max']}\n\n"
        
        axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 2].set_title('Statistik RGB')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        return fig
    
    def separate_channels(self):
        """Pisahkan dan tampilkan setiap channel RGB"""
        if self.image_rgb is None:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0, 0].imshow(self.image_rgb)
        axes[0, 0].set_title('Gambar Asli')
        axes[0, 0].axis('off')
        
        red_channel = np.zeros_like(self.image_rgb)
        red_channel[:, :, 0] = self.image_rgb[:, :, 0]
        axes[0, 1].imshow(red_channel)
        axes[0, 1].set_title('Channel Red')
        axes[0, 1].axis('off')
        
        green_channel = np.zeros_like(self.image_rgb)
        green_channel[:, :, 1] = self.image_rgb[:, :, 1]
        axes[1, 0].imshow(green_channel)
        axes[1, 0].set_title('Channel Green')
        axes[1, 0].axis('off')
        
        blue_channel = np.zeros_like(self.image_rgb)
        blue_channel[:, :, 2] = self.image_rgb[:, :, 2]
        axes[1, 1].imshow(blue_channel)
        axes[1, 1].set_title('Channel Blue')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        return fig
    
    def color_quantization(self, k=8):
        """Reduksi warna menggunakan K-Means clustering"""
        if self.image_rgb is None:
            return None
        
        data = self.image_rgb.reshape((-1, 3))
        data = np.float32(data)
        
     
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
     
        centers = np.uint8(centers)
        quantized_data = centers[labels.flatten()]
        quantized_image = quantized_data.reshape(self.image_rgb.shape)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(self.image_rgb)
        axes[0].set_title('Gambar Asli')
        axes[0].axis('off')
        
        axes[1].imshow(quantized_image)
        axes[1].set_title(f'Quantized ({k} warna)')
        axes[1].axis('off')
        
        plt.tight_layout()
        return fig, quantized_image
    
    def extract_dominant_colors(self, k=5):
        """Extract warna dominan dari gambar"""
        if self.image_rgb is None:
            return None
        
        data = self.image_rgb.reshape((-1, 3))
        data = np.float32(data)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        unique, counts = np.unique(labels, return_counts=True)
        percentages = counts / len(labels) * 100
        
        sorted_indices = np.argsort(percentages)[::-1]
        
        dominant_colors = []
        for i in sorted_indices:
            color = centers[i].astype(int)
            percentage = percentages[i]
            dominant_colors.append({
                'color': tuple(color),
                'percentage': percentage,
                'hex': f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}'
            })
        
        return dominant_colors
    
    def create_color_palette(self, dominant_colors):
        """Buat visualisasi palette warna dominan"""
        if not dominant_colors:
            return None
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        axes[0].imshow(self.image_rgb)
        axes[0].set_title('Gambar Asli')
        axes[0].axis('off')
        
        palette_height = 100
        palette_width = 500
        palette = np.zeros((palette_height, palette_width, 3), dtype=np.uint8)
        
        x_start = 0
        for color_info in dominant_colors:
            color = color_info['color']
            width = int(palette_width * color_info['percentage'] / 100)
            palette[:, x_start:x_start+width] = color
            x_start += width
        
        axes[1].imshow(palette)
        axes[1].set_title('Palette Warna Dominan')
        axes[1].axis('off')
        
        info_text = ""
        for i, color_info in enumerate(dominant_colors):
            info_text += f"Warna {i+1}: RGB{color_info['color']} ({color_info['percentage']:.1f}%)\n"
            info_text += f"         HEX: {color_info['hex']}\n"
        
        plt.figtext(0.02, 0.02, info_text, fontsize=10, family='monospace')
        plt.tight_layout()
        return fig
    
    def interactive_pixel_picker(self):
        """Mode interaktif untuk mengambil nilai RGB dengan klik mouse"""
        if self.image_rgb is None:
            print("❌ Belum ada gambar yang dimuat!")
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(self.image_rgb)
        ax.set_title('Klik pada gambar untuk mendapatkan nilai RGB')
        
        def onclick(event):
            if event.inaxes != ax:
                return
            
            x, y = int(event.xdata), int(event.ydata)
            rgb = self.get_pixel_rgb(x, y)
            
            if rgb:
                r, g, b = rgb
                print(f"📍 Koordinat: ({x}, {y})")
                print(f"🎨 RGB: ({r}, {g}, {b})")
                print(f"🔗 HEX: #{r:02x}{g:02x}{b:02x}")
                print("-" * 30)
                
                # Update title dengan nilai RGB
                ax.set_title(f'RGB: ({r}, {g}, {b}) | HEX: #{r:02x}{g:02x}{b:02x}')
                fig.canvas.draw()
        
        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
    
    def print_statistics(self):
        """Print statistik lengkap gambar"""
        if self.image_rgb is None:
            print("❌ Belum ada gambar yang dimuat!")
            return
        
        stats = self.analyze_color_channels()
        
        print("=" * 50)
        print("📊 STATISTIK GAMBAR")
        print("=" * 50)
        print(f"📁 File: {self.filename}")
        print(f"📐 Ukuran: {self.image_rgb.shape[1]} x {self.image_rgb.shape[0]} pixels")
        print(f"💾 Total pixels: {self.image_rgb.shape[0] * self.image_rgb.shape[1]:,}")
        print()
        
        for channel, data in stats.items():
            print(f"🔴🟢🔵 Channel {channel}:")
            print(f"  Mean:   {data['mean']:.2f}")
            print(f"  Std:    {data['std']:.2f}")
            print(f"  Min:    {data['min']}")
            print(f"  Max:    {data['max']}")
            print(f"  Median: {data['median']:.2f}")
            print()

# Fungsi utama untuk demo
def main():
    analyzer = RGBImageAnalyzer()
    
    print("🖼️  RGB IMAGE ANALYZER - PENGOLAHAN CITRA DIGITAL")
    print("=" * 60)
    
    if not analyzer.load_image():
        print("❌ Tidak ada gambar yang dipilih!")
        return
    
    analyzer.print_statistics()
    
    while True:
        print("\n🔧 PILIH OPERASI:")
        print("1. Tampilkan histogram RGB")
        print("2. Pisahkan channel RGB")
        print("3. Color quantization")
        print("4. Extract warna dominan")
        print("5. Mode interaktif (klik untuk RGB)")
        print("6. Ambil RGB koordinat tertentu")
        print("7. Ambil RGB dari region")
        print("8. Load gambar baru")
        print("0. Keluar")
        
        try:
            pilihan = input("\n👉 Masukkan pilihan (0-8): ")
            
            if pilihan == '1':
                fig = analyzer.create_histogram()
                if fig:
                    plt.show()
            
            elif pilihan == '2':
                fig = analyzer.separate_channels()
                if fig:
                    plt.show()
            
            elif pilihan == '3':
                k = int(input("Masukkan jumlah warna (default=8): ") or "8")
                result = analyzer.color_quantization(k)
                if result:
                    fig, quantized = result
                    plt.show()
            
            elif pilihan == '4':
                k = int(input("Masukkan jumlah warna dominan (default=5): ") or "5")
                dominant_colors = analyzer.extract_dominant_colors(k)
                if dominant_colors:
                    print("\n🎨 WARNA DOMINAN:")
                    for i, color_info in enumerate(dominant_colors):
                        print(f"{i+1}. RGB{color_info['color']} - {color_info['percentage']:.1f}% - {color_info['hex']}")
                    
                    fig = analyzer.create_color_palette(dominant_colors)
                    if fig:
                        plt.show()
            
            elif pilihan == '5':
                analyzer.interactive_pixel_picker()
            
            elif pilihan == '6':
                x = int(input("Masukkan koordinat X: "))
                y = int(input("Masukkan koordinat Y: "))
                rgb = analyzer.get_pixel_rgb(x, y)
                if rgb:
                    r, g, b = rgb
                    print(f"🎨 RGB pada ({x}, {y}): ({r}, {g}, {b})")
                    print(f"🔗 HEX: #{r:02x}{g:02x}{b:02x}")
                else:
                    print("❌ Koordinat tidak valid!")
            
            elif pilihan == '7':
                x1 = int(input("Masukkan X1: "))
                y1 = int(input("Masukkan Y1: "))
                x2 = int(input("Masukkan X2: "))
                y2 = int(input("Masukkan Y2: "))
                
                region = analyzer.get_region_rgb(x1, y1, x2, y2)
                if region is not None:
                    print(f"📐 Region size: {region.shape}")
                    print(f"🎨 Mean RGB: ({np.mean(region[:,:,0]):.1f}, {np.mean(region[:,:,1]):.1f}, {np.mean(region[:,:,2]):.1f})")
                    
                    # Tampilkan region
                    plt.figure(figsize=(8, 6))
                    plt.imshow(region)
                    plt.title(f'Region ({x1},{y1}) to ({x2},{y2})')
                    plt.axis('off')
                    plt.show()
                else:
                    print("❌ Region tidak valid!")
            
            elif pilihan == '8':
                if analyzer.load_image():
                    analyzer.print_statistics()
            
            elif pilihan == '0':
                print("👋 Terima kasih!")
                break
            
            else:
                print("❌ Pilihan tidak valid!")
                
        except ValueError:
            print("❌ Input tidak valid!")
        except KeyboardInterrupt:
            print("\n👋 Program dihentikan!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()