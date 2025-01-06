# Phuong phap dung thresh_hold
from __future__ import print_function
import cv2 as cv

# Đọc ảnh đầu vào
image_path = r'Path to image'
image = cv.imread(image_path)
cv.imshow('Original Image', image)

src = cv.imread(image_path, cv.IMREAD_GRAYSCALE)  # Đọc ảnh ở dạng grayscale
if src is None:
    print('Could not open or find the image:', image_path)
    exit(0)
# Hiển thị ảnh gốc
cv.imshow('Grayscale Image', src)
# Áp dụng lọc Gaussian để giảm nhiễu
blurred = cv.GaussianBlur(src, (5, 5), 0)
cv.imshow('Blurred Image', blurred)

# Áp dụng threshold thủ công để chọn vùng tối
# threshold_value = 70 # Giá trị ngưỡng (thay đổi phù hợp với ảnh)
threshold_value = 56.99
_, thresholded_manual = cv.threshold(blurred, threshold_value, 255, cv.THRESH_BINARY_INV)
cv.imshow('Thresholded Image - Manual', thresholded_manual)
_, thresholded = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
cv.imshow('Thresholded Image (Otsu)', thresholded)

contours_manual, _ = cv.findContours(thresholded_manual, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
output_manual = cv.cvtColor(src, cv.COLOR_GRAY2BGR)
cv.drawContours(output_manual, contours_manual, -1, (0, 255, 0), 2)
cv.imshow('Tumor Segmentation - Manual Threshold', output_manual)
# Tìm đường viền của vùng khối u
contours, _ = cv.findContours(thresholded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
output = cv.cvtColor(src, cv.COLOR_GRAY2BGR)
cv.drawContours(output, contours, -1, (0, 255, 0), 2)  # Vẽ đường viền màu xanh lá
cv.imshow('Tumor Segmentation (Otsu', output)
# Đợi người dùng nhấn phím bất kỳ và đóng cửa sổ
cv.waitKey(0)
cv.destroyAllWindows()