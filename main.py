import face_recognition
import numpy as np
import os
import cv2

# Inisialisasi kamera
camera = cv2.VideoCapture(0)
camera.set(3, 640)  # Lebar
camera.set(4, 480)  # Tinggi
camera.set(cv2.CAP_PROP_FPS, 30)

cv2.namedWindow('Face Recognition', cv2.WINDOW_NORMAL)
cv2.moveWindow('Face Recognition', 380, 100)

# List untuk menyimpan encoding dan nama wajah yang dikenal
known_faces_encodings = []
known_faces_names = []

# Path folder wajah yang dikenal
face_dir = "known_faces"

# Load semua wajah dari subfolder di "known_faces/"
for person_name in os.listdir(face_dir):
    person_folder = os.path.join(face_dir, person_name)

    if os.path.isdir(person_folder):  # Pastikan ini adalah folder
        for filename in os.listdir(person_folder):
            # Abaikan file tersembunyi seperti .DS_Store
            if filename.startswith('.'):
                continue

            filepath = os.path.join(person_folder, filename)

            try:
                image = face_recognition.load_image_file(filepath)
                encodings = face_recognition.face_encodings(image)

                if encodings:
                    known_faces_encodings.append(
                        encodings[0])  # Simpan encoding
                    # Gunakan nama folder sebagai nama orang
                    known_faces_names.append(person_name)

            except Exception as e:
                print(f"❌ Gagal membaca {filename}: {e}")


if not known_faces_encodings:
    print("⚠️ Tidak ada wajah yang dikenali! Semua wajah akan dianggap 'Unknown'.")


def face_detect(frame):
    """Deteksi wajah dalam frame"""
    if frame is None:
        print("❌ Frame kosong!")
        return []

    # Resize frame untuk mempercepat deteksi
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Deteksi wajah
    face_locations = face_recognition.face_locations(rgb_frame)
    if not face_locations:
        return []

    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    detected_faces = []

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(
            known_faces_encodings, face_encoding)
        name = "Yaki tidak bernama"

        if True in matches:
            matched_index = np.argmin(face_recognition.face_distance(
                known_faces_encodings, face_encoding))
            name = known_faces_names[matched_index]

        # Skalakan kembali lokasi wajah ke ukuran asli
        top, right, bottom, left = [val * 2 for val in face_location]
        detected_faces.append((top, right, bottom, left, name))

    return detected_faces


def box_frame(frame):
    """Gambar kotak di sekitar wajah dan beri label nama"""
    frame = cv2.flip(frame, 1)  # Membalik kamera agar tidak mirror

    for top, right, bottom, left, name in face_detect(frame):
        # Gambar kotak wajah
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Set posisi teks di tengah bawah wajah
        text_x = left + (right - left) // 2
        text_y = bottom + 25  # Geser sedikit ke bawah wajah

        # Gunakan font sederhana tanpa background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.9
        font_thickness = 2
        text_size, _ = cv2.getTextSize(name, font, font_scale, font_thickness)

        # Geser teks agar rata tengah
        text_x -= text_size[0] // 2

        # Tambahkan teks nama dengan warna biru
        cv2.putText(frame, name, (text_x, text_y), font,
                    font_scale, (255, 0, 0), font_thickness)

    return frame


def close_window():
    """Tutup kamera dan jendela OpenCV"""
    camera.release()
    cv2.destroyAllWindows()
    exit()


def main():
    print("Face Recognition is Running...")
    """Loop utama untuk menangkap frame dan melakukan deteksi wajah"""
    while True:
        ret, frame = camera.read()
        if not ret:
            print("❌ Gagal membaca frame!")
            continue

        frame = box_frame(frame)
        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            close_window()


if __name__ == "__main__":
    main()
