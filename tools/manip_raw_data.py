import os
import json
import cv2

def merge_raw(dataset_root, split_name, new_folder_name):
    os.makedirs(f"{dataset_root}/image/{split_name}/{new_folder_name}", exist_ok=True)
    for file_name in os.listdir(f"{dataset_root}/image/{split_name}/deei6_l"):
        img_l = cv2.imread(f"{dataset_root}/image/{split_name}/deei6_l/{file_name}", cv2.IMREAD_GRAYSCALE)
        img_h = cv2.imread(f"{dataset_root}/image/{split_name}/deei6_h/{file_name}", cv2.IMREAD_GRAYSCALE)
        img_z = cv2.imread(f"{dataset_root}/image/{split_name}/deei6_z/{file_name}", cv2.IMREAD_GRAYSCALE)
        merged_img = cv2.merge((img_z,img_l,img_h))
        cv2.imwrite(f"{dataset_root}/image/{split_name}/{new_folder_name}/" + file_name, merged_img)
    num_images = len(os.listdir(f"{dataset_root}/image/{split_name}/{new_folder_name}"))
    print(f"Successfully merged {num_images} images")


dataset_root = "../datasets/deei6"
merge_raw(dataset_root, "test", "deei6_merged_raw")