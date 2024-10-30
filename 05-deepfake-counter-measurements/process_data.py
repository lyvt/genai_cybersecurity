import os
import shutil
from sklearn.model_selection import train_test_split


def train_test_val_data(src_dir, dest_dir, test_size=0.2, val_size=0.1, seed=42):
    def copy_files(files, dir):
        os.makedirs(dir, exist_ok=True)
        for f in files:
            shutil.copy(f, dir)

    for label in ['fake', 'real']:
        img_list = [os.path.join(src_dir, label, img)
                    for img in os.listdir(os.path.join(src_dir, label))
                    if img.endswith(('.jpg', '.png'))]

        # Split into training and temp (validation + test)
        train_imgs, temp_imgs = train_test_split(img_list, test_size=test_size, random_state=seed)

        # Split temp into validation and test
        val_size_adjusted = val_size / (1 - test_size)  # Adjust val_size to account for temp size
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=val_size_adjusted, random_state=seed)

        # Copy files to respective directories
        copy_files(train_imgs, os.path.join(dest_dir, 'train', label))
        copy_files(val_imgs, os.path.join(dest_dir, 'val', label))
        copy_files(test_imgs, os.path.join(dest_dir, 'test', label))

    print(f"Data split complete. Train/val/test sets saved in {dest_dir}.")


if __name__ == '__main__':
    src_dir = 'data'
    dest_dir = 'data'
    train_test_val_data(src_dir, dest_dir)
