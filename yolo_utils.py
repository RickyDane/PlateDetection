from ultralytics import YOLO
import cv2, os, shutil
import pytesseract as tess
import easyocr
from datetime import datetime

model = YOLO("models/mdPlateBest.pt")
reader = easyocr.Reader(['en'])

def ResetRuns():
    path = "runs/detect/"
    try:
        shutil.rmtree(path)
        print("Directory removed successfully")
    except OSError as o:
        print(f"Error, {o.strerror}: {path}")

def PredictImage(image):
    plateText = ""
    try:
        if image is not None:
            dtStart = datetime.now()
            results = model.predict(image, conf=0.8, save=True, device="mps", verbose=False, save_crop = True)
            print("Prediction took: ", datetime.now()-dtStart)
            for result in results:
                boxCls = result.boxes[0].cls
                if boxCls == 0:
                    try:
                        box = result.boxes[0].xyxy
                        cropImage = image[int(box[0][1]):int(box[0][3]), int(box[0][0]):int(box[0][2])]
                        cropImage = cv2.resize(cropImage, (0, 0), fx=3, fy=3, interpolation=cv2.INTER_LANCZOS4)
                        threshold, cropImage = cv2.threshold(cropImage, 100, 255, cv2.THRESH_BINARY)
                        cropImage = cv2.cvtColor(cropImage, cv2.COLOR_BGR2GRAY)
                        threshold, cropImage = cv2.threshold(cropImage, 10, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                            
                        cv2.imwrite("cropped_image.jpg", cropImage)
                            
                        # plateText = tess.image_to_string(cropImage, config="-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ").replace(' ', '').upper()
                        # print(f"PyTesseract: '{plateText}'")
                            
                        plateText = reader.readtext(cropImage, detail=0, paragraph=True, allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")[0].replace(' ', '').upper()
                        print(f"EasyOCR: '{plateText}'")
                    except Exception as ex:
                        print("No result: ", ex)
                else:
                    print("No result: ", boxCls)
    except Exception as ex:
        print("Error: ", ex)
        
    if plateText != "" and plateText is not None:
        return plateText
    else:
        return "No plate detected"