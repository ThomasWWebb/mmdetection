import json
import os

def split_val_ann_from_train(dataset_root):
  val_ann_data = {}
  with open(dataset_root + "annotations/instances_train2017.json") as train_ann_file:

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
        if image_dict["file_name"] in os.listdir(dataset_root + "val2017/"):
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
      

      with open(dataset_root + "annotations/instances_val2017.json", 'w') as val_ann_file:
        json.dump(val_ann_data, val_ann_file)
      with open(dataset_root + "annotations/instances_train2017.json", 'w') as new_train_ann_file:
        json.dump(train_ann_data, new_train_ann_file)

dataset_root = "../datasets/sixray/"
split_val_ann_from_train(dataset_root)