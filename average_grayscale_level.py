import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
# input_folder = "F:/MMOTU_DS2Net-main/data/OTU-2D-Dataset-main/8_layers/Chocolate_Cyst/edited_images"
# input_folder = "F:/MMOTU_DS2Net-main/data/OTU-2D-Dataset-main/8_layers/High-grade_Serous_Cystadenoma/edited_images"
input_folder = 'images'
mask_folder = 'annotations'
# Get name of all the image files in the input folder
file_names = [f for f in os.listdir(input_folder) if f.endswith('.JPG')]
avr_list = []
contrast_list = []
count = 0
for file_name in file_names:
    # Input image path
    input_image_path = os.path.join(input_folder, file_name)
    # Open the input image
    image = cv2.imread(input_image_path)
    #Convert the image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #Find the non-zero coordinates of the mask image
    mask_image_path = os.path.join(mask_folder, file_name.replace('.JPG', '.PNG'))
    mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
    non_zero_coords = np.where(mask_image > 0)
    # Create an empty image to hold the masked part (same size as original)
    masked_image = np.zeros_like(image) # zeros
    masked_image[non_zero_coords] = image[non_zero_coords]
    #calculate the average grayscale level of the non-zero coordinates in iput image
    average_grayscale_level = np.mean(masked_image[non_zero_coords])
    # std = standard deviation (độ lệch chuẩn) 
    contrast = np.std(masked_image[non_zero_coords])
    # the higher the contrast is, the more details in the image and the more the roughness of the image

    print(f"Average grayscale level of {file_name}: {average_grayscale_level} - Contrast: {contrast}")  
    avr_list.append(average_grayscale_level)
    contrast_list.append(contrast)
    # count +=1
    # break

# print(f"Average grayscale {avr} level of all images: {avr/len(file_names)}")

# Visualize the contrast result with histogram
# n, bins, patches = plt.hist(contrast_list, bins=10, edgecolor='black', color='blue')
# # Thêm số chính xác phía trên mỗi thanh
# for i in range(len(patches)):
#     height = n[i]  # chiều cao của mỗi thanh (frequency)
#     plt.text(patches[i].get_x() + patches[i].get_width() / 2, height,  # vị trí trung tâm thanh
#              f'{int(height)}', ha='center', va='bottom')  # nội dung số
# plt.hist(contrast_list, bins=10, color='blue', edgecolor='black', alpha=0.7)
# plt.title("Histogram of Contrast Values")
# plt.xlabel("Contrast")
# plt.ylabel("Frequency")
# # Cố định các mốc trên trục x và trục y
# plt.xlim(0, 80)
# plt.ylim(0, 110)
# plt.grid(True)
# plt.show()

# Visualize the average grayscale level result with histogram
n, bins, patches = plt.hist(avr_list, bins=10, edgecolor='black', color='blue')
# Thêm số chính xác phía trên mỗi thanh
for i in range(len(patches)):
    height = n[i]  # chiều cao của mỗi thanh (frequency)
    plt.text(patches[i].get_x() + patches[i].get_width() / 2, height,  # vị trí trung tâm thanh
             f'{int(height)}', ha='center', va='bottom')  # nội dung số
plt.xticks(bins)  # Đặt các nhãn x cho các bin

plt.xlabel('Average Grayscale Level')
plt.ylabel('Frequency')
# Cố định các mốc trên trục x và trục y
plt.xlim(0, 240)
# khoảng cách giữa mỗi mốc của trục x là 20
plt.xticks(np.arange(0, 201, 20))
plt.ylim(0, 90)
plt.title('Histogram of Average Grayscale Level Values')
plt.grid(True)
plt.show()

