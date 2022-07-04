import cv2
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
smile_detector = cv2.CascadeClassifier("haarcascade_smile.xml")

webcam  = cv2.VideoCapture(0)
while True:
    succesful_frame_read, frame = webcam.read()
    grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coordinates = face_detector.detectMultiScale(grayscale_img)
    #Manual Machine Learning Right Here in Line 11
    #smile_coordinates = smile_detector.detectMultiScale(grayscale_img,scaleFactor = 1.7, minNeighbors = 20)
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (234, 23, 123), 2)
        the_face = frame[y:y+h, x:x+w]
        grayscale_img2 = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)
        smiles = smile_detector.detectMultiScale(grayscale_img2, scaleFactor=1.7, minNeighbors=20)
        #for (x_, y_, w_, h_) in smiles:
            #cv2.rectangle(the_face, (x_, y_), (x_ + w_, y_ + h_), (24, 223, 123), 2)
        if len(smiles) > 0:
            cv2.putText(frame,'Smiling',(x,y+h+40), fontScale=3, fontFace=cv2.FONT_HERSHEY_PLAIN, color = (256,256,256))
    cv2.imshow("Smile Please....",frame)
    key = cv2.waitKey(1)
    if key == 81 or key == 113:
        break
webcam.release()
cv2.destroyAllWindows()
print("Done")