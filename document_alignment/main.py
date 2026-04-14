import cv2
import numpy as np

def show_image(name, img):
    img_resized = cv2.resize(img, (600, 500))
    cv2.imshow(name, img_resized)

# Load image
img = cv2.imread("images/doc.jpeg")

if img is None:
    print("Error: Image not found!")
    exit()

show_image("Original", img)

# Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
show_image("Gray", gray)

# Blur
blur = cv2.GaussianBlur(gray, (5,5), 0)

# Edge detection
edges = cv2.Canny(blur, 50, 150)
show_image("Edges", edges)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contour_img = img.copy()
cv2.drawContours(contour_img, contours, -1, (0,255,0), 2)
show_image("Contours", contour_img)

# Find largest contour
largest = max(contours, key=cv2.contourArea)

# Approximate polygon
peri = cv2.arcLength(largest, True)
approx = cv2.approxPolyDP(largest, 0.02 * peri, True)

# Draw detected shape
doc_img = img.copy()
cv2.drawContours(doc_img, [approx], -1, (0,0,255), 3)
show_image("Detected", doc_img)

# Reorder points
def reorder(points):
    points = points.reshape((4,2))
    new_points = np.zeros((4,2), dtype=np.float32)

    add = points.sum(1)
    new_points[0] = points[np.argmin(add)]
    new_points[3] = points[np.argmax(add)]

    diff = np.diff(points, axis=1)
    new_points[1] = points[np.argmin(diff)]
    new_points[2] = points[np.argmax(diff)]

    return new_points

# Perspective transform ONLY if 4 points detected
if len(approx) == 4:
    pts1 = reorder(approx)

    width, height = 500, 700
    pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(img, matrix, (width, height))

    show_image("Warped", warped)

    # Scan effect
    scan_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, scan = cv2.threshold(scan_gray, 150, 255, cv2.THRESH_BINARY)

    show_image("Scanned", scan)

    cv2.imwrite("output/result.jpg", scan)

else:
    print("❌ Document not detected properly! Try a clearer image.")

cv2.waitKey(0)
cv2.destroyAllWindows()