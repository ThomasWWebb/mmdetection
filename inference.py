from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import mmcv
from mmcv.image import imread

import cv2
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import seaborn as sns
import os
import numpy as np
import json
import imutils
import time


def draw_bbox_pil(args, im, img, bbox_result, segm_result, CLASSES, labels, colors, bbox_thrs):
    bboxes = np.vstack(bbox_result)
    img_cp = img.copy()
    h, w, c = img.shape
    blank_image = np.zeros((h,w,c), np.uint8)
    blank_image.fill(255)
    ##seg draw
    if segm_result is not None and len(labels) > 0:  # non empty
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > args.score_thr)[0]
        np.random.seed(42)
        color_masks = [
            #np.random.randint(0, 256, (1, 3), dtype=np.uint8)
	np.array([0, 0, 255])
            for _ in range(max(labels) + 1)
        ]
        inds_mask = inds
        label_mask = labels

        
        # for i in inds:
        #     i = int(i)
        #     print('-', i)
        #     color_mask = color_masks[labels[i]]
        #     color_mask[0][0] = 0
        #     color_mask[0][1] = 0
        #     color_mask[0][2] = 0
        #     mask = segms[i].astype(bool)
        #     img[mask] = img[mask] * 256 + color_mask * 1
     

    ##bbox draw 
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
    # img = imread(img)

    if args.score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > args.score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
    
    img = np.ascontiguousarray(img)

    img_mask_list = []

    for j, (bbox, label) in enumerate(zip(bboxes, labels)):
        bbox_int = bbox.astype(np.int32)
        label_text = CLASSES[
            label] if CLASSES is not None else f'cls {label}'

        color = colors[int(label)]
        color = tuple([int(x*255) for x in color])
        
        if label_text in args.yes_cls:
            if args.crop == 'yes' and bbox_int[1] >= bbox_thrs:
                os.makedirs(f'{output_dir}/extract', exist_ok=True)
                ##crop from seg
                if segm_result is not None:
                    img_cp_mask = img_cp.copy()
                    i = int(inds_mask[j])
                    mask = segms[i].astype(bool)
                    # mask_invert = [[not x for x in list_of_bools] for list_of_bools in mask]
                    mask_invert = (np.invert(np.array(mask))).tolist() 
                    img_cp_mask[mask_invert] = 255
                    
                    img_mask_list.append(img_cp_mask)
                    
                    img_cp_mask_crp = img_cp_mask[bbox_int[1]:bbox_int[3], bbox_int[0]:bbox_int[2]]
                    cv2.imwrite(f'{output_dir}/extract/{im}_{j}.png', img_cp_mask_crp)
                    
                ##crop from bbox
                else:
                    crp_img = img[bbox_int[1]:bbox_int[3], bbox_int[0]:bbox_int[2]]
                    cv2.imwrite(f'{output_dir}/extract/{im}_{j}.png', crp_img)

            else:
                if segm_result is not None:
                    img_cp_mask = img_cp.copy()
                    i = int(inds_mask[j])
                    color_mask = color_masks[label_mask[i]]
                    # color_mask[0][0] = 0
                    # color_mask[0][1] = 0
                    # color_mask[0][2] = 0
                    # mask = np.logical_not(segms)[i].astype(bool)
                                                   
                    mask = segms[i].astype(bool)
                    img[mask] = img[mask] * 0.5 + color_mask * 0.5    
                    # img[mask] = 255
                    
                    # crp_img_mask = img[bbox_int[1]:bbox_int[3], bbox_int[0]:bbox_int[2]]

                    # crp_img_cp = img_cp[bbox_int[1]:bbox_int[3], bbox_int[0]:bbox_int[2]]
                    # cv2.imshow('mask', crp_img_mask)
                    # cv2.waitKey(0)
                    # crp_img_maskgray = cv2.cvtColor(crp_img_mask, cv2.COLOR_BGR2GRAY)
                    # crp_img_maskgray = cv2.GaussianBlur(crp_img_maskgray, (5, 5), 0) 

                    # thresh = cv2.threshold(crp_img_maskgray, 200, 255, cv2.THRESH_BINARY)[1]
                    # cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    # cnts = imutils.grab_contours(cnts)
                    # c = max(cnts, key=cv2.contourArea)
                    # x,y,w,h = cv2.boundingRect(c) 
                    # crp_img_contour = crp_img_cp[y:y+h, x:x+w]
                    # cv2.imshow('crop', crp_img_contour)
                    # cv2.waitKey(0)

                left_top = (bbox_int[0], bbox_int[1])
                right_bottom = (bbox_int[2], bbox_int[3])
                
                cv2.rectangle(
                    img, left_top, right_bottom, color=[0, 0, 255], thickness=3)
                if len(bbox) > 4:
                    label_text += f'|{bbox[-1]:.02f}'

                rectangle_bgr = (0, 0, 255)
                (text_width, text_height) = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, thickness=1)[0]
                box_coords = ((bbox_int[0] - 2,  bbox_int[1] - text_height - 2), (bbox_int[0] + text_width + 2,  bbox_int[1] + 2))
                cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
                #print(label_text)
                cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                            cv2.FONT_HERSHEY_TRIPLEX, 1, color=[255, 255, 255])
        # else:
        #     print('Invalid target class.')
    
    
    if args.crop == 'yes':
        if segm_result is not None and len(img_mask_list) > 0:
            os.makedirs(f'{output_dir}/seg',exist_ok=True)
            img_mask_final = img_mask_list[0]
            for i in img_mask_list:
                img_mask_final = cv2.bitwise_and(img_mask_final, i)
            cv2.imwrite(f'{output_dir}/seg/{im}', img_mask_final)

        else:
            cv2.imwrite(f'{output_dir}/seg/{im}', blank_image)
       
    return img

def main():

    

    parser = ArgumentParser()
    # parser.add_argument('img', help='Image file')
    # parser.add_argument('--input', help='Input image directory/image')
    # parser.add_argument('--output', help='Output image directory')
    # parser.add_argument('--db', type=str, help='dataset name')
    parser.add_argument('--yes_cls', type=str, help='Target object class, - separated [car-dog], default all classes.')
    parser.add_argument('--crop', default='no', type=str, help='Crop detected objets [yes]')
    # parser.add_argument('config', help='Config file')
    # parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.7, help='bbox score threshold')
    args = parser.parse_args()
    print(f'\n\n{args}\n\n')

    models = ["free_anchor", "cascade_rcnn"]
    backbones = ["r50"]
    datasets = ["opixray", "sixray"]
    qualities = [95]
    # model_name = "cascade_rcnn"
    # backbone = "r101"
    # dataset = "sixray"

    for model_name in models:    
        for backbone in backbones:
            for quality in qualities:
                for dataset in datasets:   
                    print("{} {} {}".format(model_name, backbone, dataset))
                    config_file = './configs/{}/{}_{}_fpn_1x_coco_{}.py'.format(model_name,model_name, backbone, dataset)
                    if model_name == "free_anchor":
                        config_file = './configs/{}/retinanet_{}_{}_fpn_1x_coco_{}.py'.format(model_name,model_name, backbone, dataset)
                    checkpoint_file = "./{}_{}_e30_{}_trueResolution/epoch_10.pth".format(model_name, backbone, dataset)
                    test_set_root = "../datasets/{}_{}".format(dataset, quality)
                    output_dir = "./{}_{}_e30_{}_trueResolution/test_detect_images_original_to_{}".format(model_name, backbone, dataset, quality)

                    WINDOW_NAME = 'Detection'
                    CLASSES = []
                    test_set_path = test_set_root + "/annotations/instances_test2017.json"
                    with open(test_set_path) as f:
                        json_data = json.load(f)
                    for data_id, data_info in json_data.items():
                        if data_id == 'categories':
                            for cat in data_info:
                                CLASSES.append(cat['name'])
                    # CLASSES = tuple(CLASSES)
                    colors = sns.color_palette("husl", len(CLASSES))

                    bbox_thrs = 0
                    args.yes_cls = None
                    if args.yes_cls is not None:
                        args.yes_cls = args.yes_cls.split("-")
                    else:
                        args.yes_cls = CLASSES

                    # build the model from a config file and a checkpoint file
                    model = init_detector(config_file, checkpoint_file)#, device=args.device)
                
                    if output_dir:
                        os.makedirs(output_dir, exist_ok=True)

                    test_set_path = test_set_root + "/test2017"
                    if os.path.isdir(test_set_path):
                        inference_times = []
                        for im in os.listdir(test_set_path):
                            print('Image: ', im)

                            #record fps
                            start = time.time()
                            result = inference_detector(model, f'{test_set_path}/{im}')
                            end = time.time()
                            inference_times.append(end - start)
                            
                            
                            #show_result_pyplot(model, f'{test_set_path}/{im}', output_dir, result, score_thr=args.score_thr)

                            ####
                            
                            img = mmcv.imread(f'{test_set_path}/{im}')
                            # img = img.copy()
                            if isinstance(result, tuple):
                                bbox_result, segm_result = result
                                if isinstance(segm_result, tuple):
                                    segm_result = segm_result[0]  # ms rcnn
                            else:
                                bbox_result, segm_result = result, None
                            # bboxes = np.vstack(bbox_result)
                            labels = [
                                np.full(bbox.shape[0], i, dtype=np.int32)
                                for i, bbox in enumerate(bbox_result)
                            ]
                            labels = np.concatenate(labels)

                            img = draw_bbox_pil(
                                args,
                                im, 
                                img, 
                                bbox_result, 
                                segm_result, 
                                CLASSES, 
                                labels, 
                                colors, 
                                bbox_thrs
                            )

                            if output_dir and args.crop != 'yes':
                                cv2.imwrite(f'{output_dir}/{im}',img)
                            else:
                                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                                cv2.imshow(WINDOW_NAME, img)
                                if cv2.waitKey(0) == 27:
                                    exit()
                        avg_inference_time = sum(inference_times) / len(inference_times)
                        print("The average inference time over {} images is {} seconds so the FPS is {}".format(len(inference_times), avg_inference_time, 1/avg_inference_time))
                    else:
                        im = test_set_path
                        print('Image: ', im)
                        result = inference_detector(model, im)
                        ####
                        img = mmcv.imread(im)
                        if isinstance(result, tuple):
                            bbox_result, segm_result = result
                            if isinstance(segm_result, tuple):
                                segm_result = segm_result[0]  # ms rcnn
                        else:
                            bbox_result, segm_result = result, None
                        
                        labels = [
                            np.full(bbox.shape[0], i, dtype=np.int32)
                            for i, bbox in enumerate(bbox_result)
                        ]
                        labels = np.concatenate(labels)

                        img = draw_bbox_pil(
                            args,
                            im, 
                            img, 
                            bbox_result, 
                            segm_result, 
                            CLASSES, 
                            labels, 
                            colors, 
                            bbox_thrs
                        )

                        if output_dir and args.crop != 'yes':
                            cv2.imwrite(f'{output_dir}/{im}',img)
                        else:
                            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                            cv2.imshow(WINDOW_NAME, img)
                            if cv2.waitKey(0) == 27:
                                exit()

if __name__ == '__main__':
    main()
