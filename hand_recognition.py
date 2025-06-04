import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


class PalmRecognizer:
    def __init__(self):
        # Konfigurasi
        self.RECOGNITION_THRESHOLD = 0.2  # Threshold untuk pengenalan

        # Inisialisasi kamera
        self.camera = cv2.VideoCapture(0)
        self.camera.set(3, 640)  # Lebar
        self.camera.set(4, 480)  # Tinggi
        self.camera.set(cv2.CAP_PROP_FPS, 30)

        # Setup window
        cv2.namedWindow('Palm Recognition', cv2.WINDOW_NORMAL)
        cv2.moveWindow('Palm Recognition', 380, 100)

        # Load custom model
        # Ganti dengan path model Anda
        self.model = load_model('palm_recognition_model.h5')
        self.input_size = (224, 224)  # Sesuaikan dengan input size model Anda

        # Database tangan yang dikenal
        self.known_hands_names = []

        # Load label names (sesuaikan dengan cara Anda menyimpan label)
        self.load_label_names("known_hands")

    def load_label_names(self, hand_dir):
        """Memuat nama label dari direktori"""
        print("Memuat label tangan yang dikenal...")

        self.known_hands_names = [name for name in os.listdir(hand_dir)
                                  if os.path.isdir(os.path.join(hand_dir, name))]

        if not self.known_hands_names:
            print("⚠️ Tidak ada label tangan yang ditemukan!")
        else:
            print(
                f"✅ Label berhasil dimuat: {len(self.known_hands_names)} kelas")

    def preprocess_image(self, frame):
        """Mempersiapkan gambar untuk model dalam grayscale"""
        # Konversi ke grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Resize dan normalisasi
        img = cv2.resize(gray, self.input_size)
        img = img_to_array(img)
        img = img / 255.0
        # Tambahkan dimensi batch dan channel
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=-1)  # Shape akhir: (1, 224, 224, 1)
        return img

    def recognize_hand(self, frame):
        """Mengenali tangan dalam frame menggunakan model custom"""
        try:
            # Preprocess frame
            processed_img = self.preprocess_image(frame)

            # Prediksi
            predictions = self.model.predict(processed_img)
            confidence = np.max(predictions)
            class_idx = np.argmax(predictions)

            if confidence > self.RECOGNITION_THRESHOLD and self.known_hands_names:
                name = self.known_hands_names[class_idx]
            else:
                name = "Unknown"

            # Mengembalikan format yang sama dengan versi MediaPipe
            return [(None, name, confidence)]

        except Exception as e:
            print(f"❌ Gagal melakukan prediksi: {str(e)}")
            return []

    def draw_hand_info(self, frame):
        """Menggambar informasi tangan"""
        frame = cv2.flip(frame, 1)
        detected_hands = self.recognize_hand(frame)

        for _, name, confidence in detected_hands:
            # Posisi teks (tengah atas frame)
            x = frame.shape[1] // 2
            y = 30

            # Format teks
            display_text = f"{name} ({confidence:.2f})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            font_thickness = 2

            # Hitung ukuran teks untuk background
            text_size, _ = cv2.getTextSize(
                display_text, font, font_scale, font_thickness)
            text_x = x - text_size[0] // 2

            # Gambar background teks
            cv2.rectangle(frame,
                          (text_x - 5, y - text_size[1] - 5),
                          (text_x + text_size[0] + 5, y + 5),
                          (50, 50, 50), -1)

            # Gambar teks
            cv2.putText(frame, display_text,
                        (text_x, y), font,
                        font_scale, (255, 255, 255), font_thickness)

        return frame

    def run(self):
        """Loop utama aplikasi"""
        print("Palm Recognition is Running...")

        try:
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    print("❌ Gagal membaca frame!")
                    continue

                frame = self.draw_hand_info(frame)
                cv2.imshow('Palm Recognition', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.close()

    def close(self):
        """Membersihkan resource"""
        self.camera.release()
        cv2.destroyAllWindows()
        print("Aplikasi ditutup")


if __name__ == "__main__":
    recognizer = PalmRecognizer()
    recognizer.run()
