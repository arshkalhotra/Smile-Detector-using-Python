import cv2


#face and smile classifiers
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')

webcam = cv2.VideoCapture(0)

while True:
    #read the current frame from thee webcam video stream
    successful_read, frame = webcam.read()
    
    if not successful_read:
         break
    
    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_detector.detectMultiScale(grayscaled_frame, scaleFactor = 1.2, minNeighbors = 10)
    
    
    #print(faces)


    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (100, 200, 50), 4)
        #the_face = frame[y:y+h , x:x+w]
        #the_face = (x, y, w, h)

        face_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        smiles = smile_detector.detectMultiScale (grayscaled_frame, scaleFactor = 1.7, minNeighbors = 20)  #manual ML
        if len(smiles) > 0:
            cv2.putText(frame, 'Smiling', (x, y+h+40), fontScale=3, fontFace=cv2.FONT_HERSHEY_PLAIN, color = (255, 255, 255))
        #find all smiles in the face
        #for (x_, y_, w_, h_) in smiles:
            #cv2.rectangle(frame, (x_,y_), (x_+w_, y_+h_), (50, 50, 200), 4)
            

    cv2.imshow('Why so serious?', frame)

    key = cv2.waitKey(1)
    if key==81 or key==113:
        break


#cleanup
webcam.release()
cv2.destroyAllWindows()
print("Code Completed")