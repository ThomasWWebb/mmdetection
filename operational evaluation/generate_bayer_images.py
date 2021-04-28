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

def add_red(img, img_red):
    for height_index in range(1, len(img), 2):
        for row_index in range(1, len(img[0]), 2):
            img[height_index, row_index] = img_red[height_index, row_index]
    return  img

def add_green(img, img_green):
    for height_index in range(0, len(img), 2):
        for row_index in range(1, len(img[0]), 2):
            img[height_index, row_index] = img_green[height_index, row_index]
        for row_index in range(0, len(img[0]), 2):
            img[height_index+1, row_index] = img_green[height_index+1, row_index]
    return img

def bayer_pattern(dataset_root, split_name, new_folder_name):
    os.makedirs(f"{dataset_root}/image/{split_name}/{new_folder_name}", exist_ok=True)
    for file_name in os.listdir(f"{dataset_root}/image/{split_name}/deei6_l"):
        img_l = cv2.imread(f"{dataset_root}/image/{split_name}/deei6_l/{file_name}", cv2.IMREAD_GRAYSCALE)
        img_h = cv2.imread(f"{dataset_root}/image/{split_name}/deei6_h/{file_name}", cv2.IMREAD_GRAYSCALE)
        img_z = cv2.imread(f"{dataset_root}/image/{split_name}/deei6_z/{file_name}", cv2.IMREAD_GRAYSCALE)
        img_l = add_red(img_l, img_z)
        img_l = add_green(img_h, img_z)
        cv2.imwrite(f"{dataset_root}/image/{split_name}/{new_folder_name}/" + file_name, img_l)
    num_images = len(os.listdir(f"{dataset_root}/image/{split_name}/{new_folder_name}"))
    print(f"Successfully produced {num_images} images")

dataset_root = "../../datasets/deei6"
bayer_pattern(dataset_root, "test", "deei6_bayer(z,h,l)")