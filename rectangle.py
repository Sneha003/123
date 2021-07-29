import cv2
picture=cv2.imread("galaxy.jpg")
print(picture.shape)
cv2.rectangle(picture,(10,10),(100,100),(18,98,226),5)
cv2.rectangle(picture,(120,120),(150,150),(18,98,226),-3)
cv2.imshow("window1" ,picture)
cv2.waitKey(0)
picture.release()
cv2.destroyAllWindows()

