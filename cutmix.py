import json
import sys
import cv2
from datetime import date
import random
import os
import glob

# im1 = cv2.imread('gilardoni-fep-me640amx-20201211-112136-00011566-service-01-color.png')
# h1, w1, _ = im1.shape
# im2 = cv2.imread('gilardoni-fep-me640amx-20201211-112431-00011572-service-01-color.png')
# h2, w2, _ = im2.shape

# cv2.imshow('Show', im1)
# cv2.waitKey(0)

# cv2.imshow('Show', im2)
# cv2.waitKey(0)

# crop_im1 = im1[100:150, 90:200]
# crop_im2 = im2[200:250, 300:310]
# c2_h2, c2_w2, _ = crop_im2.shape

# im1[100:150, 90:200] = im2[200:250, 300:410]
# cv2.imshow('Show', im1)
# cv2.waitKey(0)

# im1[100:100+c2_h2, 90:90+c2_w2] = crop_im2
# cv2.imshow('Show', im1)
# cv2.waitKey(0)

json_file = '/home/neel/data/cnn_model/dataset/OPIXray/annotation/OPIXray_train_index1.json'
img_dir = '/home/neel/data/cnn_model/dataset/OPIXray/image/train'
img_out = '/home/neel/data/cnn_model/dataset/OPIXray/image/train-cutmix'
OUTPUT_FILENAME = '/home/neel/data/cnn_model/dataset/OPIXray/annotation/OPIXray_train_index1-aug_cutmix.json'

os.makedirs(img_out, exist_ok=True)

files = glob.glob(img_out + '/*')
for f in files:
    os.remove(f)

today = date.today()
db_name = 'OPIXray'
output = {
    "info": {
        "description": db_name,
        "url": "https://www.durham.ac.uk",
        "version": "1.0",
        "year": date.today().year,
        "contributor": "DU",
        "date_created": today.strftime("%d/%m/%Y")
    },
    "licenses": [{
        "url": "https://www.durham.ac.uk",
        "id": 0,
        "name": "DU, Research work"
    }],
    "images": [],
    "annotations": [],
    "categories": []
}

out_cat = []
output_1 = {
    "images": [],
    "annotations": [],
}

output_2 = {
    "images": [],
    "annotations": [],
}

out_cat.append(output_1)
out_cat.append(output_2)

# print(out_cat[0]["images"])


with open(json_file) as f:
    json_info = json.load(f)


output['images'] = json_info['images']
output['annotations'] = json_info['annotations']
output['categories'] = json_info['categories']

f_img_id = json_info['images'][-1]["id"]
f_ann_id = json_info['annotations'][-1]["id"]

# extract category
yes_catid = [2,4]
for i, cat in enumerate(yes_catid):
    
    image_id_old = []
    image_id_new = []
    annotation_id = 1
    ann_img_id = 1
    img_id = 1

    for ann in json_info['annotations']:
        if ann['category_id'] == cat:
            if ann['image_id'] not in image_id_old:
                image_id_old.append(ann['image_id'])
                
                for im in json_info['images']:
                    if ann['image_id'] == im['id']:
                        
                        out_cat[i]['images'].append({
                            "id": im["id"],
                            "width": im["width"],
                            "height": im["height"],
                            "file_name": im["file_name"],
                            "license": 'for_science'
                        })
                     
                # ann_img_id +=1

            out_cat[i]['annotations'].append({
                "segmentation": ann['segmentation'],
                "iscrowd": 0,
                "image_id": ann['image_id'],
                "category_id": cat,
                "id": ann['id'],
                "bbox": ann['bbox'],
                "area": ann['area'],
                "trunc": [],
                "occlu": []
            })  
            # annotation_id +=1


# cutmix
per_sample = 0.5  # change the % of total images to be used
mixr = 0.5 # 50% of the object

l_cat1 = len(out_cat[0]['images'])
l_cat2 = len(out_cat[1]['images'])

if l_cat1 < l_cat2:
    rand_cat1 = random.sample(range(0, l_cat1), int(l_cat1*per_sample))
    rand_cat2 = random.sample(range(0, l_cat2), int(l_cat2*((1/l_cat2) * int(l_cat1*per_sample)) ) )
if l_cat1 > l_cat2: 
    rand_cat1 = random.sample(range(0, l_cat1), int(l_cat1* ((100/l_cat1) * int(l_cat2*per_sample))  ))
    rand_cat2 = random.sample(range(0, l_cat2), int(l_cat2*per_sample))
if l_cat1 == l_cat2:
    rand_cat1 = random.sample(range(0, l_cat1), int(l_cat1*per_sample))
    rand_cat1 = random.sample(range(0, l_cat2), int(l_cat2*per_sample))

print(l_cat1, l_cat2, len(rand_cat1), len(rand_cat2))

for i in rand_cat1:
    f_img_id +=1
    f_ann_id +=1

    img1 = out_cat[0]["images"][i]
    img2 = out_cat[1]["images"][i]
    
    img_id1 = img1["id"]
    img_id2 = img2["id"] 

    img_f1 = cv2.imread(f'{img_dir}/{img1["file_name"]}')
    h1, w1, _ = img_f1.shape
    img_f2 = cv2.imread(f'{img_dir}/{img2["file_name"]}')
    h2, w2, _ = img_f2.shape
   
    for ann in out_cat[0]["annotations"]:
        if img_id1 == ann["image_id"]:
            bbox1 = ann["bbox"]
            cat1 = ann["category_id"]

    for ann in out_cat[1]["annotations"]:
        if img_id2 == ann["image_id"]:
            bbox2 = ann["bbox"]
            cat2 = ann["category_id"]

    print('=====')
    print(img1["file_name"], bbox1)
    # cv2.rectangle(img_f1, (int(bbox1[0]), int(bbox1[1])), (int(bbox1[0] + bbox1[2]), int(bbox1[1] + bbox1[3])), (0,0,255), 1)
    # cv2.imshow('Show1', img_f1)
    # cv2.waitKey(0)
    
    print(img2["file_name"], bbox2)
    # cv2.rectangle(img_f2, (int(bbox2[0]), int(bbox2[1])), (int(bbox2[0] + bbox2[2]), int(bbox2[1] + bbox2[3])), (0,0,255), 1)    
    # cv2.imshow('Show2', img_f2)
    # cv2.waitKey(0)
    print('=====')

    crop_im1 = img_f1[int(bbox1[1]):int(bbox1[1]+bbox1[3]), int(bbox1[0]):int(bbox1[0]+bbox1[2]*mixr)]
    c1_h1, c1_w1, _ = crop_im1.shape
    
    crop_im2 = img_f2[int(bbox2[1]):int(bbox2[1]+bbox2[3]), int(bbox2[0]):int(bbox2[0]+bbox2[2])]
    c2_h2, c2_w2, _ = crop_im2.shape

    # print('----')
    # print('crop_im1: ', c1_h1, c1_w1)
    # cv2.imshow('crop_im1', crop_im1)
    # cv2.waitKey(0)
    
    # print('crop_im2: ', c2_h2, c2_w2)
    # cv2.imshow('crop_im2', crop_im2)
    # cv2.waitKey(0)
    # print('----')

    crop_im1 = cv2.resize(crop_im1, (int(c2_w2/2),int(c2_h2)), interpolation=cv2.INTER_AREA)
    c1_h1, c1_w1, _ = crop_im1.shape
    # print('resize crp img1: ', c1_h1, c1_w1)
    # cv2.imshow('crop_im1_resize', crop_im1)
    # cv2.waitKey(0)

    img_f2[int(bbox2[1]):int(bbox2[1]+c1_h1), int(bbox2[0]):int(bbox2[0]+c1_w1)] = crop_im1

    crop_bbox = []
    crop_bbox.append([ int(bbox2[0]), int(bbox2[1]), c1_w1, c1_h1 ] )
    crop_bbox.append([ int(bbox2[0]+ c2_w2/2), int(bbox2[1]), int(bbox2[2]- (c2_w2/2)), int(bbox2[3])] )

    # print(crop_bbox1, crop_bbox2)

    cut_file_name = f'{img_out}/{img1["file_name"]}_{img1["file_name"]}_{cat1}-{cat2}.png'
    cv2.imwrite(cut_file_name, img_f2)    

    # cv2.rectangle(img_f2, (crop_bbox[0][0], crop_bbox[0][1]), (crop_bbox[0][0] + crop_bbox[0][2], crop_bbox[0][1] + crop_bbox[0][3]), (0,0,255), 1)
    # cv2.rectangle(img_f2, (crop_bbox[1][0], crop_bbox[1][1]), (crop_bbox[1][0] + crop_bbox[1][2], crop_bbox[1][1] + crop_bbox[1][3]), (255,0,0), 1)
    # cv2.imshow('CutMix', img_f2)
    # cv2.waitKey(0)
    cut_j_f = f'{img1["file_name"]}_{img1["file_name"]}_{cat1}-{cat2}.png'
    output['images'].append({
        "id": f_img_id,
        "width": w2,
        "height": h2,
        "file_name": cut_j_f,
        "license": 'for_science'
    })

    for i, c in enumerate([cat1, cat2]):
        output['annotations'].append({
            "segmentation": [],
            "iscrowd": 0,
            "image_id": f_img_id,
            "category_id": c,
            "id": f_ann_id,
            "bbox": crop_bbox[i],
            "area": crop_bbox[i][2]*crop_bbox[i][3],
            "trunc": [],
            "occlu": []
        })

        f_ann_id +=1

# json operation
# for i, oc in enumerate(out_cat):
#     out_cat[i] = json.dumps(out_cat[i])

# print(out_cat[0])
# print('=======================================')
# print(out_cat[1])


json_output = json.dumps(output)
with open(OUTPUT_FILENAME, 'w') as f:
    f.write(json_output)