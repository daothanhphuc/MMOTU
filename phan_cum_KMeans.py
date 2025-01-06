# Phuong phap K-Means
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
# Đọc ảnh đầu vào
image_path = r'Path to the image'
# image_path  = r'F:\phucdz.jpg'
src = cv.imread(image_path)
if src is None:
    print("Could not open or find the image.")
    exit(0)
# Chuyển đổi ảnh sang không gian màu L*a*b* (tùy chọn, giúp phân vùng màu tốt hơn)
src_lab = cv.cvtColor(src, cv.COLOR_BGR2Lab)

pixel_values = src_lab.reshape((-1, 3))  # Chuyển đổi ảnh 3D thành mảng 2D (H*W, 3)
pixel_values = np.float32(pixel_values)  # Chuyển thành kiểu float32 cho K-means

# Số lượng nhóm (clusters)
k = 5  # Số lượng vùng cần phân vùng 
# Tiêu chí dừng cho thuật toán K-means
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 200, 0.1)
# Áp dụng K-means
_, labels, centers = cv.kmeans(pixel_values, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
# Chuyển đổi tâm (centroid) về kiểu uint8
centers = np.uint8(centers)
# Gán nhãn pixel thành màu của tâm tương ứng
segmented_image = centers[labels.flatten()]

# Đưa ảnh trở lại định dạng ban đầu (H, W, C)
segmented_image = segmented_image.reshape(src_lab.shape)
# Hiển thị ảnh phân vùng
segmented_image_bgr = cv.cvtColor(segmented_image, cv.COLOR_Lab2BGR)
# cv.imshow('Original Image', src)
# cv.imshow('Segmented Image (K-means)', segmented_image_bgr)
# Sử dụng mathplotlib để hiển thị ảnh
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(src, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(segmented_image_bgr, cmap='gray')
plt.title('k = 2')

plt.tight_layout()
plt.show()

# Tách riêng một vùng cụ thể (ví dụ: nhóm 1)
mask = (labels.flatten() == 1)  # Tạo mask cho vùng nhóm 1
masked_image = np.copy(src)
masked_image = masked_image.reshape((-1, 3))
masked_image[~mask] = [0, 0, 0]  # Gán nền (background) về đen
masked_image = masked_image.reshape(src.shape)
# Hiển thị vùng đã tách
cv.imshow('Extracted Region (Cluster 1)', masked_image)

cv.waitKey(0)
cv.destroyAllWindows()