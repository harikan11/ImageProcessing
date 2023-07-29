import cv2
import pytesseract
import numpy as np

pytesseract.pytesseract.tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe"


#img = cv2.imread(r"C:\Users\Harika Naishadham\OneDrive\Documents\IP\sqa-vision\trial.jpg", cv2.IMREAD_GRAYSCALE)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.imread(r'C:\Users\Harika Naishadham\OneDrive\Documens\IP\sqa-vision\trial.jpg', cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


texts = pytesseract.image_to_string(img)
noise = cv2.medianBlur(img, 3)
#thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

kernel = np.ones((3, 3), np.uint8)
thresh = cv2.threshold(noise, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
dilated = cv2.dilate(thresh, kernel, iterations=1)
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



min_contour_area = 100
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
rois = []
for cnt in filtered_contours:
    x, y, w, h = cv2.boundingRect(cnt)
    roi = img[y : y + h, x : x + w]
    rois.append(roi)

config = "-l eng — oem 1 — psm 6"

def correct_skew(image, delta=1, limit=5):
    def determine_score(arr, angle):
        data = img.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return histogram, score

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    corrected = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )

    return best_angle, corrected


clean = thresh.copy()



boxes = pytesseract.image_to_boxes(img)

print(boxes)
def draw_boxes_on_character(img):
    img_width = img.shape[1]
    img_height = img.shape[0]
    boxes = pytesseract.image_to_boxes(img)

    for box in boxes.splitlines():
        box = box.split(" ")
        character = box[0]
        x = int(box[1])
        y = int(box[2])
        x2 = int(box[3])
        y2 = int(box[4])
        cv2.rectangle(img, (x, img_height - y), (x2, img_height - y2), (0, 255, 0), 1)
 
        cv2.putText(img, character, (x, img_height -y2), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
    
    return img

 
img = draw_boxes_on_character(img)

raw_data = pytesseract.image_to_data(img)
print(raw_data )
print(texts)
cv2.waitKey(0)