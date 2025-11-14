import cv2, os, argparse, time

def ensure(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def capture_images(name, count=30, delay=0.1):
    dataset_dir = os.path.join("dataset", name)
    ensure(dataset_dir)
    cam = cv2.VideoCapture(0)
    i = 0
    print("Press 'c' to capture, 'q' to quit.")

    while i < count:
        ret, frame = cam.read()
        if not ret:
            print("Camera read error.")
            break

        cv2.imshow("Enroll - Press 'c' to capture", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            filename = os.path.join(dataset_dir, f"{int(time.time())}_{i}.jpg")
            cv2.imwrite(filename, frame)
            print("Saved:", filename)
            i += 1
        elif key == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"Captured {i} images for {name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture face images")
    parser.add_argument("--name", required=True)
    parser.add_argument("--count", type=int, default=30)
    args = parser.parse_args()
    capture_images(args.name, args.count)
