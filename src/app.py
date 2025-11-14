from flask import Flask, request, render_template_string
import face_recognition, json, numpy as np
from pathlib import Path
from PIL import Image
import base64, io

app = Flask(__name__)
ENC_FILE = "encodings.json"

TEMPLATE = """
<h1>Face Recognition Web Demo</h1>
<form method="post" enctype="multipart/form-data">
  <input type="file" name="file">
  <input type="submit" value="Upload">
</form>
{% if result %}
<h3>Result: {{ result }}</h3>
<img src="data:image/jpeg;base64,{{ imgdata }}">
{% endif %}
"""

def load_encodings():
    if not Path(ENC_FILE).exists():
        return [], []
    data = json.load(open(ENC_FILE))
    return [np.array(e) for e in data["encodings"]], data["names"]

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    img_b64 = None
    known_encs, known_names = load_encodings()

    if request.method == "POST":
        file = request.files["file"]
        img = Image.open(file.stream).convert("RGB")
        img_arr = np.array(img)

        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        boxes = face_recognition.face_locations(img_arr)

        if boxes:
            encs = face_recognition.face_encodings(img_arr, boxes)
            predictions = []

            for enc in encs:
                match = None
                for i, known in enumerate(known_encs):
                    if face_recognition.compare_faces([known], enc)[0]:
                        match = known_names[i]
                        break
                predictions.append(match or "Unknown")

            result = ", ".join(predictions)
        else:
            result = "No face found."

    return render_template_string(TEMPLATE, result=result, imgdata=img_b64)
