from flask import Flask, request, render_template, redirect
from yolo_utils import *
import torch

app = Flask(__name__)

@app.route("/")
def main():
    return render_template("upload_image.html")

@app.route("/api/plate-detection", methods=["POST", "GET"])
def main_app_run():
    if request.method == "POST":
        file = request.files["image"]
        if file is not None:
            file.save("current_process.jpg")
            file = cv2.imread("current_process.jpg")
            ResetRuns()
            result = PredictImage(file)

    return render_template("upload_image.html", plateDetection=result)

@app.route("/api/train-plate-detection", methods=["POST", "GET"])
def train_dataset():
    if request.method == "POST":
        torch.cuda.empty_cache()
        try:
            ResetRuns()
        except Exception as ex:
            print("No run reset possible: ", ex)
        YOLO("models/yolov8l.pt").train(data="mdPlate.yaml", save=True, workers=0, epochs=12000, imgsz=480, batch=10, patience=300, device="0", val=False, cache=True, verbose=False, amp=False)
        if shutil.copy("runs/detect/train/weights/best.pt", "models/mdPlateBest.pt"):
            print("Model saved to models/mdPlateBest.pt")
    return redirect("/")

if __name__ == "__main__":
    app.run(debug=True, port=8081)