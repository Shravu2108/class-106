import cv2

img = cv2.imread("boy.jpg")

gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 

faces = face_cascade.detectMultiScale(gray, 1.1, 3)

print(len(faces))

for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0) , 2)

    # cropping the face's image and storing it into a seperate file
    roi = img[y:y+h, x:x+w]
    cv2.imwrite("Face.png", roi)
    

cv2.imshow("image" , img)

cv2.waitKey(0)


# -----------------------------------------------------------------------------------

# detectMultiscale()

# ● scaleFactor : This parameter sets the percentage
# amount to reduce the size of the detection
# window(it is the portion of the image used for
# detection features) after every detection

# This means the detection window size gets smaller
# and smaller after every round of detection, by the
# value chosen for scaleFactor to increase precision.
# Increasing the scaleFactor, helps to increase the
# detection accuracy.

# Possible range between 1.1 to 1.9

# ● minNeighbors: Parameter specifying how many
# facial features that need to be present, to detect the
# face.

# To get more accurate result let us set values for the
# parameters scalefactor & minNeighbors in
# detectMultiScale()
