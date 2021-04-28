from PIL import Image
import os

def compress_file(file_path, new_file_path, quality):
    im1 = Image.open(file_path)
    im1.save(new_file_path, quality=quality)

def compress_folder_contents(folder_path, new_folder_path, quality):
    print("Compressing {} at {} quality".format(folder_path, quality))
    os.makedirs(new_folder_path, exist_ok=True)
    file_names = os.listdir(folder_path)
    for file_name in file_names:
        compress_file(folder_path + file_name, new_folder_path + file_name, quality)

def compress_coco_dataset(root_path, dataset_name, quality_arr):
    folders = ["train2017", "val2017", "test2017"]
    for quality in quality_arr:
        for folder in folders:
            compress_folder_contents(f"{root_path}/{dataset_name}/{folder}/", f"{root_path}/{dataset_name}_{quality}/{folder}/", quality)