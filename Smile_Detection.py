import cv2
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
smile_detector = cv2.CascadeClassifier("haarcascade_smile.xml")

webcam  = cv2.VideoCapture(0)
while True:
    succesful_frame_read, frame = webcam.read()
    grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coordinates = face_detector.detectMultiScale(grayscale_img)
    #Manual Machine Learning Right Here in Line 11
    smile_coordinates = smile_detector.detectMultiScale(grayscale_img,scaleFactor = 1.7, minNeighbors = 20)
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (234, 23, 123), 2)
    for (x, y, w, h) in smile_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (24, 223, 123), 2)
    cv2.imshow("Smile Please....",frame)
    key = cv2.waitKey(1)
    if key == 81 or key == 113:
        break
webcam.release()
cv2.destroyAllWindows()
print("Done")