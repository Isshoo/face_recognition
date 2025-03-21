import face_recognition
import numpy as np
import os
import cv2


face_ref = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

camera = cv2.VideoCapture(0)
camera.set(3, 640)  # Lebar
camera.set(4, 480)  # Tinggi


def face_detect(frame):
    if frame is None:
        print("❌ Frame kosong!")
        return []

    # Kecilkan gambar sebelum diproses
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)
    if not face_locations:
        print("⚠️ Tidak ada wajah terdeteksi.")
        return []

    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    detected_faces = []

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(
            known_faces_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            matched_index = np.argmin(face_recognition.face_distance(
                known_faces_encodings, face_encoding))
            name = known_faces_names[matched_index]

        # Sesuaikan kembali ukuran lokasi wajah setelah resize
        top, right, bottom, left = [val * 2 for val in face_location]
        detected_faces.append((top, right, bottom, left, name))

    return detected_faces


def box_frame(frame):
    for top, right, bottom, left, name in face_detect(frame):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


def close_window():
    camera.release()
    cv2.destroyAllWindows()
    exit()


known_faces_encodings = []
known_faces_names = []

# Load semua wajah yang dikenal dari folder known_faces/
face_dir = "known_faces"

for filename in os.listdir(face_dir):
    filepath = os.path.join(face_dir, filename)

    # Baca gambar dan ambil encoding wajah
    image = face_recognition.load_image_file(filepath)
    encoding = face_recognition.face_encodings(image)

    if encoding:  # Pastikan ada wajah dalam gambar
        known_faces_encodings.append(encoding[0])
        # Simpan nama file tanpa ekstensi
        known_faces_names.append(os.path.splitext(filename)[0])


def main():
    while True:
        ret, frame = camera.read()
        if not ret:
            print("❌ Gagal membaca frame!")
            continue

        box_frame(frame)
        cv2.imshow('Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            close_window()


if __name__ == "__main__":
    main()
