import cv2

# Load Haar Cascade for car detection
car_cascade = cv2.CascadeClassifier('haarcascades/cars.xml')

# Function to detect cars in each frame
def detect_car(img):
    car_img = img.copy()
    car_rects = car_cascade.detectMultiScale(car_img,scaleFactor=1.15,minNeighbors=5)

    # Draw rectangles around detected cars
    for (x, y, w, h) in car_rects:
        cv2.rectangle(car_img, (x, y), (x+w, y+h), (255, 0, 255), 3)
    return car_img

# Open video file
cap = cv2.VideoCapture('videos/test_video.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video finished")
        break
    
    frame = detect_car(frame)
    cv2.imshow('car detection', frame)

    # Exit when ESC key is pressed
    if cv2.waitKey(30) == 27:
        break

cap.release()
cv2.destroyAllWindows()
