# Calculate the width and height of the cropped tumor
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# input_folder = 'images'
# mask_folder = 'annotations'
# width= []
# height = []
# # Get name of all the image files in the input folder
# file_names = [f for f in os.listdir(input_folder) if f.endswith('.JPG')]
# for file_name in file_names:
#     # Đọc ảnh gốc và ảnh mặt nạ
#     input_image_path = os.path.join(input_folder, file_name)
#     image = cv2.imread(input_image_path)
#     mask_image_path = os.path.join(mask_folder, file_name.replace('.JPG', '.PNG'))
#     mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)

#     # Tìm các điểm có giá trị khác 0 (vùng khối u) trên mặt nạ
#     non_zero_coords = np.where(mask_image > 0)

#     # Tính toán các tọa độ tối thiểu và tối đa
#     x_min, x_max = np.min(non_zero_coords[1]), np.max(non_zero_coords[1])
#     y_min, y_max = np.min(non_zero_coords[0]), np.max(non_zero_coords[0])
#     print(f"Position of {file_name}: ({x_min}, {y_min}) - ({x_max}, {y_max})")

#     # Cắt ảnh gốc dựa trên tọa độ tìm được
#     cropped_image = image[y_min:y_max, x_min:x_max]
#     print(f"Width x Height of {file_name}: {cropped_image.shape[1]} x {cropped_image.shape[0]}")
#     width.append(cropped_image.shape[1])
#     height.append(cropped_image.shape[0])
#     # cv2.imshow("Cropped Tumor", cropped_image)
#     # cv2.waitKey(0)
#     # break

# # Hiển thị các ảnh đã cắt theo từng class
# plt.figure(figsize=(6, 6))
# plt.title("Size of Cropped Tumor") 
# # cố định hệ trục x và y
# plt.xlim(0, 800)
# plt.xticks(np.arange(0, 701, 100))
# plt.ylim(0, 700)
# plt.yticks(np.arange(0, 701, 100))
# plt.plot(width, height, 'ro') # 'ro' is red circle
# plt.xlabel('Width')
# plt.ylabel('Height')
# plt.show()


# Display the width and height of the cropped tumor of every class in a scatter plot
def process_images(image_dir, mask_dir):
    widths = []
    heights = []
    filenames = [f for f in os.listdir(image_dir) if f.endswith('.JPG')]

    for file_name in filenames:
        # Read the input image and mask
        input_image_path = os.path.join(image_dir, file_name)
        image = cv2.imread(input_image_path)
        mask_image_path = os.path.join(mask_dir, file_name.replace('.JPG', '.PNG'))
        mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)

        non_zero_coords = np.where(mask_image > 0)
        # if len(non_zero_coords[0]) == 0:  # if no tumor region found, skip the image
        #     continue
        x_min, x_max = np.min(non_zero_coords[1]), np.max(non_zero_coords[1])
        y_min, y_max = np.min(non_zero_coords[0]), np.max(non_zero_coords[0])
        cropped_image = image[y_min:y_max, x_min:x_max]

        # Store the width and height 
        widths.append(cropped_image.shape[1])
        heights.append(cropped_image.shape[0])
    return widths, heights

classes = {
    "Chocolate Cyst": {
        "image_dir": 'F:/MMOTU_DS2Net-main/data/OTU-2D-Dataset-main/8_layers/Chocolate_Cyst/images',
        "mask_dir": 'F:/MMOTU_DS2Net-main/data/OTU-2D-Dataset-main/8_layers/Chocolate_Cyst/annotations'
    },
    "High-grade Serous Cystadenoma": {
        "image_dir": 'F:/MMOTU_DS2Net-main/data/OTU-2D-Dataset-main/8_layers/High-grade_Serous_Cystadenoma/images',
        "mask_dir": 'F:/MMOTU_DS2Net-main/data/OTU-2D-Dataset-main/8_layers/High-grade_Serous_Cystadenoma/annotations'
    },
    "Mucinous Cystadenoma": {
        "image_dir": 'F:/MMOTU_DS2Net-main/data/OTU-2D-Dataset-main/8_layers/Mucinous_Cystadenoma/images',
        "mask_dir": 'F:/MMOTU_DS2Net-main/data/OTU-2D-Dataset-main/8_layers/Mucinous_Cystadenoma/annotations'
    },
    "Ovary Normal": {
        "image_dir": 'F:/MMOTU_DS2Net-main/data/OTU-2D-Dataset-main/8_layers/Ovary_Normal/images',
        "mask_dir": 'F:/MMOTU_DS2Net-main/data/OTU-2D-Dataset-main/8_layers/Ovary_Normal/annotations'
    },
    "Serous Cystadenoma": {
        "image_dir": 'F:/MMOTU_DS2Net-main/data/OTU-2D-Dataset-main/8_layers/Serous_Cystadenoma/images',
        "mask_dir": 'F:/MMOTU_DS2Net-main/data/OTU-2D-Dataset-main/8_layers/Serous_Cystadenoma/annotations'
    },
    "Simple Cyst": {
        "image_dir": 'F:/MMOTU_DS2Net-main/data/OTU-2D-Dataset-main/8_layers/Simple_Cyst/images',
        "mask_dir": 'F:/MMOTU_DS2Net-main/data/OTU-2D-Dataset-main/8_layers/Simple_Cyst/annotations'
    },
    "Theca Cell Tumor": {
        "image_dir": 'F:/MMOTU_DS2Net-main/data/OTU-2D-Dataset-main/8_layers/Theca_Cell_Tumor/images',
        "mask_dir": 'F:/MMOTU_DS2Net-main/data/OTU-2D-Dataset-main/8_layers/Theca_Cell_Tumor/annotations'
    }
}

colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange']
plt.figure(figsize=(10, 8))
for i, (class_name, paths) in enumerate(classes.items()):
    widths, heights = process_images(paths['image_dir'], paths['mask_dir'])
    plt.scatter(widths, heights, color=colors[i], label=class_name)

# Visualize the width vs height of tumor regions for different classes
plt.xlabel('Width')
plt.ylabel('Height')
plt.xlim(0, 800)
plt.xticks(np.arange(0, 701, 100))
plt.ylim(0, 700)
plt.yticks(np.arange(0, 701, 100))
plt.title('Width vs Height of Tumor Regions for Different Classes')
plt.legend()
plt.show()






