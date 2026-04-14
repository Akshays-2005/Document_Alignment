import cv2

img = cv2.imread("images/rolls.jpg")

cv2.imshow("Test Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()