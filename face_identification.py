import cv2

face_cascade = cv2.CascadeClassifier('C:\\Users\\user\\Anaconda3\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')

#img = cv2.imread('hero.jpeg')
img = cv2.imread('women.jpg')
print(img.shape)
print(img.size)
print(img.dtype)

thumb = cv2.resize(img,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC)
cv2.imshow('img',thumb)
cv2.imwrite('thumb.png',thumb)
cv2.waitKey(0)

grad = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(grad,1.1,4)

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow('img',img)
cv2.imwrite('reseult.png',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


