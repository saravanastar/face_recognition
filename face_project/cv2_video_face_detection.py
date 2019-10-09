import cv2
import numpy as np
from face_project import IdentifyFace as face_detect
import PIL.Image


cap = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )
    for (x, y, w, h) in faces:
        face_image = frame[y:y + h, x:x + w]
        # cv2.imshow('cut_frame', face_image)
        person = face_detect.identiyPerson(face_image)
        print(person)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, person, (x + h, int(y+w)), cv2.FONT_HERSHEY_SIMPLEX, 1 , (0, 255, 0), 2, cv2.LINE_AA)

    # Display the resulting frame
    out.write(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()