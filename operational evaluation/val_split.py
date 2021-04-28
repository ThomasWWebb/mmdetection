import os
import json
import math
import random
   
def get_class_counts(dataset_root, ann_file_name, class_min, class_max):
    with open(dataset_root + "annotations/" + ann_file_name) as train_ann_file:
        class_counts = {}
        for class_num in range(class_min, class_max + 1):
            class_counts[class_num] = 0
        train_ann_data = json.load(train_ann_file)
        for ann_dict in train_ann_data["annotations"]:
            class_counts[ann_dict["category_id"]] += 1
        for key in class_counts:
            class_counts[key] = round(class_counts[key] / len(train_ann_data["annotations"]) * 100, 2)
        return class_counts

def check_single_object_per_image(dataset_root, ann_file_name):
    with open(dataset_root + "annotations/" + ann_file_name) as train_ann_file:
        train_ann_data = json.load(train_ann_file)
        image_id_list = []
        for ann_dict in train_ann_data["annotations"]:
            image_id_list.append(ann_dict["image_id"])
        if len(image_id_list) != len(set(image_id_list)):
            return False
    return True

def split_into_classes(dataset_root, ann_file_name, class_min, class_max):
    with open(dataset_root + "annotations/" + ann_file_name) as train_ann_file:
        class_splits = {}
        for class_num in range(class_min, class_max + 1):
            class_splits[class_num] = []
        train_ann_data = json.load(train_ann_file)
        for ann_dict in train_ann_data["annotations"]:
            class_splits[ann_dict["category_id"]].append(ann_dict)
    return class_splits

def get_val_class_splits(train_class_splits, val_proportion):
    val_class_splits = {}
    for class_num in train_class_splits:
        class_size = len(train_class_splits[class_num])
        val_class_splits[class_num] = []
        val_class_proportion = math.floor(class_size * val_proportion)
        for _ in range(val_class_proportion):
            random_index = random.choice(range(class_size - 1))
            class_size -= 1
            val_class_splits[class_num].append(train_class_splits[class_num][random_index])
            del train_class_splits[class_num][random_index]
    return val_class_splits, train_class_splits

def move_images_train_to_val(dataset_root, ann_file_name,val_class_splits):
    with open(dataset_root + "annotations/" + ann_file_name) as train_ann_file:
        train_ann_data = json.load(train_ann_file)
        for class_num in val_class_splits:
            for ann_dict in val_class_splits[class_num]:
                image_id = ann_dict["image_id"]
                for image_dict in train_ann_data["images"]:
                    if image_id == image_dict["id"]:
                        os.rename(dataset_root + "image/train" + image_dict["file_name"][1:], dataset_root + "image/val" + image_dict["file_name"][1:])


def split_val_ann_from_train(dataset_root, dataset_name):
    val_ann_data = {}
    with open(dataset_root + "annotations/" + dataset_name + "_train.json") as train_ann_file:
        new_train_images = []
        new_train_annotations = []
        train_ann_data = json.load(train_ann_file)
        print("old train set JSON with {} images and {} annotations".format(len(train_ann_data["images"]), len(train_ann_data["annotations"])))
        val_ann_data["info"] = train_ann_data["info"]
        val_ann_data["licenses"] = train_ann_data["licenses"]

        for index in range(len(train_ann_data["images"])):
            image_dict = train_ann_data["images"][index]
            selected_image_ann = []
            for ann_dict in train_ann_data["annotations"]:
                if ann_dict["image_id"] == image_dict["id"]:
                    selected_image_ann.append(ann_dict)
            if image_dict["file_name"][10:] in os.listdir(dataset_root + "image/val/" + dataset_name):
                if "images" not in val_ann_data:
                    val_ann_data["images"] = [image_dict]
                    val_ann_data["annotations"] = selected_image_ann
                else:
                    val_ann_data["images"].append(image_dict)
                    val_ann_data["annotations"] += selected_image_ann
            else:
                new_train_images.append(image_dict)
                new_train_annotations += selected_image_ann

        val_ann_data["categories"] = train_ann_data["categories"]
        print(val_ann_data.keys())
        train_ann_data["images"] = new_train_images
        train_ann_data["annotations"] = new_train_annotations
        print("Validation set JSON with {} images and {} annotations".format(len(val_ann_data["images"]), len(val_ann_data["annotations"])))
        print("New train set JSON with {} images and {} annotations".format(len(train_ann_data["images"]), len(train_ann_data["annotations"])))
        

        with open(dataset_root + "annotations/" + dataset_name + "_val.json", 'w') as val_ann_file:
            json.dump(val_ann_data, val_ann_file)
        with open(dataset_root + "annotations/" + dataset_name + "_train.json", 'w') as new_train_ann_file:
            json.dump(train_ann_data, new_train_ann_file)

def copy_val_split(dataset_root, existing_val_dir, train_dir):
    print(f'train images: {len(os.listdir(os.path.join(dataset_root, "image/train", train_dir)))}')
    for img_file_name in os.listdir(os.path.join(dataset_root, "image/val", existing_val_dir)):
        os.rename(dataset_root + "image/train/" + train_dir + "/" + img_file_name, dataset_root + "image/val/" + train_dir + "/" +img_file_name)
    print(f'train images after: {len(os.listdir(os.path.join(dataset_root, "image/train", train_dir)))}')
    print(f'new val images: {len(os.listdir(os.path.join(dataset_root, "image/val", train_dir)))}')

def check_image_annotations(dataset_root, dataset_name, split_name):
    with open(dataset_root + "annotations/" + dataset_name + "_" + split_name + ".json") as ann_file:
        ann_data = json.load(ann_file)
        image_list = os.listdir(dataset_root + "image/" + split_name + "/" + dataset_name)
        print(len(image_list))
        print(len(ann_data["images"]))
        print("Missing from image folder:")
        for image in ann_data["images"]:
            if image["file_name"][10:] not in image_list:
                print(image["file_name"])


dataset_root = "../datasets/deei6/"
#ann_file_names = os.listdir(dataset_root + "annotations/")
ann_file_name = "deei6_l_train.json"
class_min = 1
class_max = 6
val_proportion = 1/7

# train_class_splits = split_into_classes(dataset_root, ann_file_name, class_min, class_max)
# val_class_splits, train_class_splits = get_val_class_splits(train_class_splits, val_proportion)
# move_images_train_to_val(dataset_root, ann_file_name,val_class_splits)
#

copy_val_split(dataset_root, "deei6_fc", "deei6_z")
split_val_ann_from_train(dataset_root, "deei6_z")
#check_image_annotations(dataset_root, "deei6_l", "val")

# for ann_file_name in ann_file_names:
#     if ann_file_name[len(ann_file_name)-5:] == ".json":
#         print(ann_file_name)
#         print(get_class_counts(dataset_root, ann_file_name, class_min, class_max))

