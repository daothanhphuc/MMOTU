import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
import os
from PIL import Image
import matplotlib.pyplot as plt

# Đọc ảnh xám
input_folder = "tumor_without_bg"
# input_folder = "images"
mask_folder = "annotations"
homogeneity_list = []
file_names = [f for f in os.listdir(input_folder) if f.endswith('.png')]
for file_name in file_names:
    input_image_path = os.path.join(input_folder, file_name)
    img = Image.open(input_image_path).convert("L") # grayscale
    mask_image_path = os.path.join(mask_folder, file_name.replace('.JPG', '.PNG'))
    mask_image = Image.open(mask_image_path).convert("L")  
    mask_array = np.array(mask_image)
    img = np.array(img).astype(np.uint8)
    # Tim cac diem co gia tri khac 0 (vung khoi u) 
    non_zero_coords = np.where(mask_array > 0)
    # Tính toán các tọa độ tối thiểu và tối đa
    x_min, x_max = np.min(non_zero_coords[1]), np.max(non_zero_coords[1])
    y_min, y_max = np.min(non_zero_coords[0]), np.max(non_zero_coords[0])
    cropped_image = img[y_min:y_max, x_min:x_max]
    cv2.imshow("image", cropped_image)
    cv2.waitKey()
    # Cal GLCM (co-occurrence matrix)
    glcm = graycomatrix(cropped_image, 
                        distances=[1], 
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                        levels=256, 
                        symmetric=True, 
                        normed=True)
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    print(f"Homogeneity and Roughness of tumor in image {file_name}: {homogeneity} ")
    homogeneity_list.append(homogeneity)
    # roughness_list.append(roughness)
    # break
# Visualize the homogeneity result with histogram
plt.figure(figsize=(8, 6))
n, bins, patches = plt.hist(homogeneity_list, bins=10, edgecolor='black', color='blue')
for i in range(len(patches)):
    height = n[i]  # chiều cao của mỗi thanh (frequency)
    plt.text(patches[i].get_x() + patches[i].get_width() / 2, height,  # vị trí trung tâm thanh
             f'{int(height)}', ha='center', va='bottom')  # nội dung số
plt.hist(homogeneity_list, bins=10, color='blue', edgecolor='black', alpha=0.7)
plt.title("Histogram of homogeneity Values")
plt.xlabel("Homogeneity")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# # # Read the original image (tumor with black regions)
# # image = cv2.imread("images/6.JPG")

# # # Read the mask image (binary mask with white tumor and black background)
# # mask = cv2.imread("annotations/6.PNG", cv2.IMREAD_GRAYSCALE)
# # # Ensure the mask is binary (just in case)
# # _, binary_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
# # # Apply the mask to the original image to keep only the tumor
# # masked_tumor = cv2.bitwise_and(image, image, mask=binary_mask)
# # # Create a 4-channel (RGBA) image by adding an alpha channel
# # b, g, r = cv2.split(masked_tumor)
# # alpha = binary_mask  # Use the binary mask as the alpha channel
# # tumor_rgba = cv2.merge([b, g, r, alpha])

# # # Save the image with a transparent background (PNG)
# # cv2.imwrite("tumor_with_transparent_bg.png", tumor_rgba)

# # # Display the result (optional)
# # cv2.imshow("Tumor with Transparent Background", tumor_rgba)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()


# import cv2
# import numpy as np
# from skimage import feature
# # Đọc ảnh nhị phân đã cắt chứa khối u
# tumor_image = cv2.imread('edited_images/6.jpg', cv2.IMREAD_GRAYSCALE)

# # Tạo mặt nạ nhị phân cho khối u bằng cách sử dụng phân ngưỡng
# mask = cv2.imread('annotations/6.PNG', cv2.IMREAD_GRAYSCALE)

# # Lọc ra các pixel thuộc vùng u (nơi mask != 0)
# tumor_pixels = tumor_image[mask != 0]
# print(tumor_pixels.size)
# ground_truth_size = np.sum(mask == 0)
# print (ground_truth_size)
# # Get the exact range of the width and height of the tumor
# non_zero_coords = np.where(mask > 0)
# x_min, x_max = np.min(non_zero_coords[1]), np.max(non_zero_coords[1])
# y_min, y_max = np.min(non_zero_coords[0]), np.max(non_zero_coords[0])
# print(f"Width of the tumor: {x_min} : {x_max}")
# print(f"Height of the tumor: {y_min} : {y_max}")

# # Tính độ lệch chuẩn (standard deviation) của các pixel thuộc khối u
# contrast = np.std(tumor_pixels)

# # In kết quả
# print(f"Contrast (standard deviation) of tumor region: {contrast}")

# # Tạo một ảnh mới với chỉ vùng khối u
# tumor_region = np.zeros_like(tumor_image)
# tumor_region[mask != 0] = tumor_image[mask != 0]

# # Tạo GLCM từ ảnh vùng u
# glcm = feature.graycomatrix(tumor_region, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

# # Tính độ tương phản từ GLCM
# contrast = feature.graycoprops(glcm, prop='contrast')[0, 0]

# # In kết quả
# print(f"Contrast (using GLCM) of tumor region: {contrast}")

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # Đọc ảnh đã cắt chứa khối u
# tumor_image = cv2.imread('6.png', cv2.IMREAD_GRAYSCALE)

# # Tạo histogram
# plt.hist(tumor_image.ravel(), bins=256, range=[0, 256])
# plt.title('Histogram of Tumor Image')
# plt.xlabel('Pixel Intensity')
# plt.ylabel('Frequency')
# plt.show()

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # Hàm tính toán histogram cho một ảnh và trả về dữ liệu
# def compute_histogram(image_path):
#     # Đọc ảnh mức xám
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     # Tính histogram với 256 bins (khoảng giá trị từ 0 đến 255)
#     hist, bins = np.histogram(image, bins=256, range=[0, 256])
#     # Chuẩn hóa histogram để biểu đồ của các ảnh khác nhau có thể so sánh được
#     hist = hist / hist.sum()
#     return hist

# # Đường dẫn đến các ảnh (thay thế đường dẫn bằng đường dẫn thực tế của bạn)
# image_paths = ['6.png', '7.png', '8.png']

# # Tạo một danh sách các màu khác nhau để phân biệt các histogram
# colors = ['blue', 'red', 'green']

# # Vẽ histogram cho nhiều ảnh trên cùng một biểu đồ
# plt.figure(figsize=(10, 6))

# # Lặp qua các ảnh và tính toán histogram
# for i, image_path in enumerate(image_paths):
#     hist = compute_histogram(image_path)
#     plt.plot(hist, color=colors[i], label=f'Image {i + 1}')  # Vẽ đường histogram với màu tương ứng

# # Thiết lập tiêu đề và các thông số của biểu đồ
# plt.title('Histograms of Multiple Images')
# plt.xlabel('Pixel Intensity')
# plt.ylabel('Normalized Frequency')
# plt.legend()  # Hiển thị chú thích để phân biệt các ảnh
# plt.show()




