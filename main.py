import cv2


face_ref = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

camera = cv2.VideoCapture(0)


def face_detect(frame):
    if frame is None:
        print("❌ Frame kosong!")
        return []

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_ref.detectMultiScale(
        gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces


def box_frame(frame):
    for x, y, w, h in face_detect(frame):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 4)


def close_window():
    camera.release()
    cv2.destroyAllWindows()
    exit()


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
