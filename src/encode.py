import face_recognition, os, json
from pathlib import Path

DATASET_DIR = "dataset"
OUTPUT_FILE = "encodings.json"

def encode_dataset():
    encodings = []
    names = []

    for person_folder in os.listdir(DATASET_DIR):
        person_path = os.path.join(DATASET_DIR, person_folder)

        if not os.path.isdir(person_path):
            continue

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)

            try:
                img = face_recognition.load_image_file(img_path)
                boxes = face_recognition.face_locations(img)

                if len(boxes) == 0:
                    print("No face found:", img_path)
                    continue

                enc = face_recognition.face_encodings(img, boxes)[0]
                encodings.append(enc.tolist())
                names.append(person_folder)
                print("Encoded:", img_path)

            except Exception as e:
                print("Error:", img_path, e)

    data = {"encodings": encodings, "names": names}
    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f)

    print("Saved encodings to", OUTPUT_FILE)

if __name__ == "__main__":
    Path(DATASET_DIR).mkdir(exist_ok=True)
    encode_dataset()
