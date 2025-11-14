import cv2, face_recognition, json, numpy as np
from pathlib import Path

ENC_FILE = "encodings.json"
TOLERANCE = 0.5

def load_encodings():
    if not Path(ENC_FILE).exists():
        print("Run encode.py first.")
        return [], []
    data = json.load(open(ENC_FILE))
    return [np.array(e) for e in data["encodings"]], data["names"]

def main():
    known_encs, known_names = load_encodings()
    if not known_encs:
        return

    cam = cv2.VideoCapture(0)
    while True:
        ret, frame = cam.read()
        if not ret:
            break

        small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb = small[:, :, ::-1]

        boxes = face_recognition.face_locations(rgb)
        encs = face_recognition.face_encodings(rgb, boxes)

        for (top, right, bottom, left), enc in zip(boxes, encs):
            matches = face_recognition.compare_faces(known_encs, enc, tolerance=TOLERANCE)
            name = "Unknown"

            if True in matches:
                idx = matches.index(True)
                name = known_names[idx]

            top *= 2; right *= 2; bottom *= 2; left *= 2
            cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
            cv2.putText(frame, name, (left, top-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
