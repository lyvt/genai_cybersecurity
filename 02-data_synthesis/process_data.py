import numpy as np  # Import the numpy library for array manipulation
import pandas as pd
import os
from PIL import Image
import shutil
import glob
import random


def binary_to_image(binary_file):
    """
    Define a function named binary_to_image that takes a binary file as input
    """
    image = np.zeros((256, 256),
                     dtype=np.uint8)  # Create an empty 256x256 image array of type uint8 (values between 0-255)

    with open(binary_file, "rb") as file:  # Open the binary file in read mode as binary
        byte_data = file.read()  # Read the content of the binary file into byte_data
        i = 9  # Start from index 9
        while i < len(byte_data):  # Iterate through the byte_data until the end
            data = byte_data[i:i + 47]  # Extract 47 bytes of data
            j = 0
            while j < len(data):  # Iterate through the data
                data_temp = data[j:j + 5]  # Extract 5 bytes of data
                x = int(data_temp[:2], 16) if "?" not in data_temp[:2].decode(
                    "utf-8") else 0  # Convert the first 2 bytes to an integer
                y = int(data_temp[3:], 16) if "?" not in data_temp[3:].decode(
                    "utf-8") else 0  # Convert the last 2 bytes to an integer
                x1 = min(x, 255)  # Ensure x is within the range of the image width
                y1 = min(y, 255)  # Ensure y is within the range of the image height
                image[x1, y1] += 1  # Increment the pixel value at position (x1, y1) in the image
                j += 6  # Move to the next 6 bytes
            i += 47 + 11  # Move to the next block of data
    return image  # Return the final image


def prepare_data(src_path, dest_path):
    labels_df = pd.read_csv('trainLabels.csv')
    for index, row in labels_df.iterrows():
        file_id = row['Id']
        class_label = row['Class']
        class_directory = os.path.join(dest_path, str(class_label))
        os.makedirs(class_directory, exist_ok=True)
        binary_file_path = os.path.join(src_path, f"{file_id}.bytes")
        if os.path.exists(binary_file_path):
            image = binary_to_image(binary_file_path)
            image_filename = os.path.join(class_directory, f"{file_id}.png")
            img = Image.fromarray(image)
            img.save(image_filename)
        else:
            print(f"Binary file {binary_file_path} does not exist.")
    print("Images have been generated and organized into folders based on their class labels.")


def train_test_split(dir_path):
    train_data = os.path.join(dir_path, 'train')
    test_data = os.path.join(dir_path, 'test')
    os.makedirs(test_data, exist_ok=True)
    class_list = os.listdir(train_data)
    for i, class_ in enumerate(class_list):
        image_list = glob.glob(os.path.join(train_data, class_, "*.png"))
        unique_num = random.sample(range(0, len(image_list)), int(len(image_list) * 0.1))
        os.makedirs(os.path.join(test_data, class_), exist_ok=True)
        for num in unique_num:
            filename = os.path.basename(image_list[num])
            shutil.move(image_list[num], os.path.join(test_data, class_, filename))


if __name__ == '__main__':
    train_test_split(dir_path='data')
