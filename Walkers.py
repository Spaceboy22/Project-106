import cv2
video=cv2.VideoCapture("walking.avi")
body_classifier=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Create our body classifier
while True:
    ret,frame=video.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    bodies=body_classifier.detectMultiScale(gray)
    for x,y,w,h in bodies:
        cv2.rectangle(frame(x,y),(x+w),(y+h))
    cv2.imshow("Video",frame)
    cv2.waitKey(0)
    
