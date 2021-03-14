import inspect
import mmcv
from mmcv.utils import build_from_cfg

import random
import numpy as np
from numpy import random
import cv2 
import matplotlib.pyplot as plt 

from mmdet.core import PolygonMasks
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from mmdet.core.bbox.iou_calculators.iou2d_calculator import BboxOverlaps2D
from mmdet.core.bbox.iou_calculators.builder import build_iou_calculator
from ..builder import PIPELINES
from .loading import (LoadImageFromFile)

#https://github.com/open-mmlab/mmdetection/blob/master/mmdet/datasets/pipelines/transforms.py
@PIPELINES.register_module()
class custom_RandomCrop(object):
    """Random crop the image & bboxes & masks.
    The absolute `crop_size` is sampled based on `crop_type` and `image_size`,
    then the cropped results are generated.
    Args:
        crop_size (tuple): The relative ratio or absolute pixels of
            height and width.
        crop_type (str, optional): one of "relative_range", "relative",
            "absolute", "absolute_range". "relative" randomly crops
            (h * crop_size[0], w * crop_size[1]) part from an input of size
            (h, w). "relative_range" uniformly samples relative crop size from
            range [crop_size[0], 1] and [crop_size[1], 1] for height and width
            respectively. "absolute" crops from an input with absolute size
            (crop_size[0], crop_size[1]). "absolute_range" uniformly samples
            crop_h in range [crop_size[0], min(h, crop_size[1])] and crop_w
            in range [crop_size[0], min(w, crop_size[1])]. Default "absolute".
        allow_negative_crop (bool, optional): Whether to allow a crop that does
            not contain any bbox area. Default False.
        bbox_clip_border (bool, optional): Whether clip the objects outside
            the border of the image. Defaults to True.
    Note:
        - If the image is smaller than the absolute crop size, return the
            original image.
        - The keys for bboxes, labels and masks must be aligned. That is,
          `gt_bboxes` corresponds to `gt_labels` and `gt_masks`, and
          `gt_bboxes_ignore` corresponds to `gt_labels_ignore` and
          `gt_masks_ignore`.
        - If the crop does not contain any gt-bbox region and
          `allow_negative_crop` is set to False, skip this image.
    """

    def __init__(self,
                 crop_size,
                 crop_type='absolute',
                 allow_negative_crop=False,
                 bbox_clip_border=True):
        if crop_type not in [
                'relative_range', 'relative', 'absolute', 'absolute_range'
        ]:
            raise ValueError(f'Invalid crop_type {crop_type}.')
        if crop_type in ['absolute', 'absolute_range']:
            assert crop_size[0] > 0 and crop_size[1] > 0
            assert isinstance(crop_size[0], int) and isinstance(
                crop_size[1], int)
        else:
            assert 0 < crop_size[0] <= 1 and 0 < crop_size[1] <= 1
        self.crop_size = crop_size
        self.crop_type = crop_type
        self.allow_negative_crop = allow_negative_crop
        self.bbox_clip_border = bbox_clip_border
        # The key correspondence from bboxes to labels and masks.
        self.bbox2label = {
            'gt_bboxes': 'gt_labels',
            'gt_bboxes_ignore': 'gt_labels_ignore'
        }
        self.bbox2mask = {
            'gt_bboxes': 'gt_masks',
            'gt_bboxes_ignore': 'gt_masks_ignore'
        }

    def _crop_data(self, results, crop_size, allow_negative_crop):
        """Function to randomly crop images, bounding boxes, masks, semantic
        segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
            crop_size (tuple): Expected absolute size after cropping, (h, w).
            allow_negative_crop (bool): Whether to allow a crop that does not
                contain any bbox area. Default to False.
        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        assert crop_size[0] > 0 and crop_size[1] > 0
        for key in results.get('img_fields', ['img']):
            img = results[key]
            margin_h = max(img.shape[0] - crop_size[0], 0)
            margin_w = max(img.shape[1] - crop_size[1], 0)
            offset_h = np.random.randint(0, margin_h + 1)
            offset_w = np.random.randint(0, margin_w + 1)
            crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

            # crop the image
            img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
            img_shape = img.shape
            results[key] = img
        results['img_shape'] = img_shape

        # crop bboxes accordingly and clip to the image boundary
        for key in results.get('bbox_fields', []):
            # e.g. gt_bboxes and gt_bboxes_ignore
            bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h],
                                   dtype=np.float32)
            bboxes = results[key] - bbox_offset
            if self.bbox_clip_border:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            valid_inds = (bboxes[:, 2] > bboxes[:, 0]) & (
                bboxes[:, 3] > bboxes[:, 1])
            # If the crop does not contain any gt-bbox area and
            # allow_negative_crop is False, skip this image.
            if (key == 'gt_bboxes' and not valid_inds.any()
                    and not allow_negative_crop):
                return None
            results[key] = bboxes[valid_inds, :]
            # label fields. e.g. gt_labels and gt_labels_ignore
            label_key = self.bbox2label.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid_inds]

            # mask fields, e.g. gt_masks and gt_masks_ignore
            mask_key = self.bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][
                    valid_inds.nonzero()[0]].crop(
                        np.asarray([crop_x1, crop_y1, crop_x2, crop_y2]))

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = results[key][crop_y1:crop_y2, crop_x1:crop_x2]

        return results

    def _get_crop_size(self, image_size):
        """Randomly generates the absolute crop size based on `crop_type` and
        `image_size`.
        Args:
            image_size (tuple): (h, w).
        Returns:
            crop_size (tuple): (crop_h, crop_w) in absolute pixels.
        """
        h, w = image_size
        if self.crop_type == 'absolute':
            return (min(self.crop_size[0], h), min(self.crop_size[1], w))
        elif self.crop_type == 'absolute_range':
            assert self.crop_size[0] <= self.crop_size[1]
            crop_h = np.random.randint(
                min(h, self.crop_size[0]),
                min(h, self.crop_size[1]) + 1)
            crop_w = np.random.randint(
                min(w, self.crop_size[0]),
                min(w, self.crop_size[1]) + 1)
            return crop_h, crop_w
        elif self.crop_type == 'relative':
            crop_h, crop_w = self.crop_size
            return int(h * crop_h + 0.5), int(w * crop_w + 0.5)
        elif self.crop_type == 'relative_range':
            crop_size = np.asarray(self.crop_size, dtype=np.float32)
            crop_h, crop_w = crop_size + np.random.rand(2) * (1 - crop_size)
            return int(h * crop_h + 0.5), int(w * crop_w + 0.5)

    def __call__(self, results):
        """Call function to randomly crop images, bounding boxes, masks,
        semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        image_size = results['img'].shape[:2]
        crop_size = self._get_crop_size(image_size)
        results = self._crop_data(results, crop_size, self.allow_negative_crop)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(crop_size={self.crop_size}, '
        repr_str += f'crop_type={self.crop_type}, '
        repr_str += f'allow_negative_crop={self.allow_negative_crop}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
        return repr_str

@PIPELINES.register_module()
class custom_CutMix(object):
    def __init__(self, cutMix_prob, class_targets=None):
        self.loadImageFromFile = build_from_cfg(dict(type='LoadImageFromFile'), PIPELINES)
        self.probability = cutMix_prob
        self.class_target = class_targets

    def __call__(self, results):
        if random.random() < self.probability:
            if (self.class_target == None) or (set(results["ann_info"]["labels"]) & set(self.class_target.keys())):
                #get the current image
                
                img_1 = results["img"]
                img_1_bboxes = results["ann_info"]["bboxes"]
                #get the data of the extra image
                extra_img = results["extra_img"]
                extra_img = self.loadImageFromFile(extra_img)
                img_2 = extra_img["img"]
                img_2_bboxes = extra_img["ann_info"]["bboxes"]
                #Get random object from the main image and extra image
                if self.class_target == None:
                    img_1_index = random.choice(range(len(img_1_bboxes)))
                    img_2_index = random.choice(range(len(img_2_bboxes)))
                else:
                    img_1_class_indices = [i for i, x in enumerate(results["ann_info"]["labels"]) if x in self.class_target.keys()]
                    img_1_index = random.choice(img_1_class_indices)
                    img_2_class_indices = [i for i, x in enumerate(extra_img["ann_info"]["labels"]) if x in self.class_target.values()]
                    img_2_index = random.choice(img_2_class_indices)

                img_1_bbox = img_1_bboxes[img_1_index]
                img_2_bbox = img_2_bboxes[img_2_index]

                img_1_object = img_1[int(img_1_bbox[1]):int(img_1_bbox[3]), int(img_1_bbox[0]):int(img_1_bbox[2])]
                img_2_object = img_2[int(img_2_bbox[1]):int(img_2_bbox[3]), int(img_2_bbox[0]):int(img_2_bbox[2])]
                img_2_object = self.resize(img_1_object, img_2_object)
                img_1_object[:, int((img_1_bbox[2] - img_1_bbox[0]) // 2):] = img_2_object[:, int((img_1_bbox[2] - img_1_bbox[0]) // 2):]
                img_1[int(img_1_bbox[1]):int(img_1_bbox[3]), int(img_1_bbox[0]):int(img_1_bbox[2])] = img_1_object

                #Combine the two images
                results["img"] = img_1
                #add the extra image bboxes and class labels to the mixed image's annotations
                img_1_bbox[2] = img_1_bbox[0] + (img_1_bbox[2] - img_1_bbox[0]) // 2
                img_2_bbox[0] = img_1_bbox[2]
                img_2_bbox[1] = img_1_bbox[1]
                img_2_bbox[2] = img_1_bbox[2] + (img_1_bbox[2] - img_1_bbox[0]) // 2
                img_2_bbox[3] = img_1_bbox[3]
                results["ann_info"]["bboxes"][img_1_index] = img_1_bbox
                results["ann_info"]["bboxes"] = np.concatenate((results["ann_info"]["bboxes"],[img_2_bbox]))
                results["ann_info"]["labels"] = np.concatenate((results["ann_info"]["labels"],[extra_img["ann_info"]["labels"][img_2_index]]))
        return results

    def resize(self, img_1_object, img_2_object):
        c1_h1, c1_w1, _ = img_1_object.shape
        img_2_object = cv2.resize(img_2_object, (int(c1_w1),int(c1_h1)), interpolation=cv2.INTER_AREA)
        return img_2_object

@PIPELINES.register_module()
class custom_MixUp(object):
    def __init__(self, mixUp_prob):
        self.loadImageFromFile = build_from_cfg(dict(type='LoadImageFromFile'), PIPELINES)
        self.probability = mixUp_prob

    def __call__(self, results):
        if random.random() < self.probability:
            #get the current image
            img_1 = results["img"]
            #get the data of the extra image
            extra_img = results["extra_img"]
            extra_img = self.loadImageFromFile(extra_img)
            img_2 = extra_img["img"]
            img_2_bboxes = extra_img["ann_info"]["bboxes"]
            #resize the extra image and its bboxes to be the same size as the current image to enable a consistent weighted addition
            img_2, img_2_bboxes = self.resize(img_2, img_2_bboxes, img_1.shape[1], img_1.shape[0])
            #Combine the two images
            mixed_img = cv2.addWeighted(img_1, 0.5, img_2, 0.5, 0.0)
            results["img"] = mixed_img
            #add the extra image bboxes and class labels to the mixed image's annotations
            results["ann_info"]["bboxes"] = np.concatenate((results["ann_info"]["bboxes"],img_2_bboxes))
            results["ann_info"]["labels"] = np.concatenate((results["ann_info"]["labels"],extra_img["ann_info"]["labels"]))
        return results

    def resize(self, img, bboxes, new_w, new_h):
        w_ratio = new_w / img.shape[1]
        h_ratio = new_h / img.shape[0]
        img = cv2.resize(img, (new_w, new_h))
        for bbox in bboxes:
            bbox[0] = bbox[0] * w_ratio
            bbox[2] = bbox[2] * w_ratio
            bbox[1] = bbox[1] * h_ratio
            bbox[3] = bbox[3] * h_ratio
        return img, bboxes

@PIPELINES.register_module()
class custom_bboxMixUp(object):
    def __init__(self, mixUp_prob):
        self.loadImageFromFile = build_from_cfg(dict(type='LoadImageFromFile'), PIPELINES)
        self.probability = mixUp_prob
        self.iou_limit = 0.3

    def __call__(self, results):
        if random.random() < self.probability:
            #get the current image
            img_1 = results["img"]
            #get the data of the extra image
            extra_img = results["extra_img"]
            extra_img = self.loadImageFromFile(extra_img)
            img_2 = extra_img["img"]
            img_2_bboxes = extra_img["ann_info"]["bboxes"]
            #resize the extra image and its bboxes to be the same size as the current image to enable a consistent mix-up
            img_2, img_2_bboxes = self.resize(img_2, img_2_bboxes, img_1.shape[1], img_1.shape[0])
            #randomly selected a bbox from the extra image
            chosen_index = random.choice(range(len(img_2_bboxes)))
            potential_bboxes = [img_2_bboxes[chosen_index]]
            img_2_bboxes = np.delete(img_2_bboxes, chosen_index, 0)
            #add all bboxes that sufficiently overlap with the randomly chosen bbox
            chosen_bboxes = self.get_overlapping_bboxes(potential_bboxes, img_2_bboxes, self.iou_limit)
            #Check that the chosen bboxes from the extra image don't overlap over some threshold with the bboxes of the current image (img_1)
            if self.no_overlaps(chosen_bboxes, results["ann_info"]["bboxes"], self.iou_limit):
                #combine all chosen bboxes into one large area
                combined_bbox = self.get_combined_bbox(chosen_bboxes)
                #Combine the two areas denoted by the combined_bbox in both images, retaining most of the extra image's bbox information
                bbox_mixed = cv2.addWeighted(img_1[combined_bbox[1]:combined_bbox[3], combined_bbox[0]:combined_bbox[2]], 0.25, img_2[combined_bbox[1]:combined_bbox[3], combined_bbox[0]:combined_bbox[2]], 0.75, 0.0)
                #Assign the combined bbox area back to the current image
                img_1[combined_bbox[1]:combined_bbox[3], combined_bbox[0]:combined_bbox[2]] = bbox_mixed
                results["img"] = img_1
                #add the extra image bboxes and class labels contained in the combined bbox to the current image's annotations
                results["ann_info"]["bboxes"] = np.concatenate((results["ann_info"]["bboxes"],chosen_bboxes))
                chosen_labels = self.get_labels(chosen_bboxes, extra_img["ann_info"]["bboxes"], extra_img["ann_info"]["labels"])
                results["ann_info"]["labels"] = np.concatenate((results["ann_info"]["labels"],chosen_labels))
        return results

    def get_labels(self, chosen_bboxes, all_bboxes, labels):
        chosen_labels = []
        chosen_bboxes = [tuple(x) for x in chosen_bboxes]
        for index in range(len(all_bboxes)):
            if tuple(all_bboxes[index]) in chosen_bboxes:
                chosen_labels.append(labels[index])
        return chosen_labels

    def get_combined_bbox(self, chosen_bboxes):
        x1_coords = []
        x2_coords = []
        y1_coords = []
        y2_coords = []
        for bbox in chosen_bboxes:
            x1_coords.append(int(bbox[0]))
            x2_coords.append(int(bbox[0]) + int(bbox[2]))
            y1_coords.append(int(bbox[1]))
            y2_coords.append(int(bbox[1]) + int(bbox[3]))
        return [min(x1_coords), min(y1_coords), max(x2_coords), max(y2_coords)]


    def no_overlaps(self, chosen_bboxes, bboxes_to_avoid, iou_limit):
        for bbox in chosen_bboxes:
            bb1 = {'x1':int(bbox[0]), 'x2':int(bbox[0]) + int(bbox[2]), 'y1':int(bbox[1]), 'y2':int(bbox[1]) + int(bbox[3])}
            for bbox_to_compare in bboxes_to_avoid:
                bb2 = {'x1':int(bbox_to_compare[0]), 'x2':int(bbox_to_compare[0]) + int(bbox_to_compare[2]), 'y1':int(bbox_to_compare[1]), 'y2':int(bbox_to_compare[1]) + int(bbox_to_compare[3])}
                if self.get_iou(bb2,bb1) > iou_limit:
                    return False
        return True

    def get_overlapping_bboxes(self, chosen_bboxes, possible_bboxes, iou_limit):
        if len(possible_bboxes) > 0:
            return self.acceptable_overlaps(chosen_bboxes, possible_bboxes, iou_limit)
        return chosen_bboxes
            
    def acceptable_overlaps(self, chosen_bboxes, bbox_list, iou_limit):
        bbox = chosen_bboxes[0] 
        bb1 = {'x1':int(bbox[0]), 'x2':int(bbox[0]) + int(bbox[2]), 'y1':int(bbox[1]), 'y2':int(bbox[1]) + int(bbox[3])}
        for bbox_to_compare in bbox_list:
            bb2 = {'x1':int(bbox_to_compare[0]), 'x2':int(bbox_to_compare[0]) + int(bbox_to_compare[2]), 'y1':int(bbox_to_compare[1]), 'y2':int(bbox_to_compare[1]) + int(bbox_to_compare[3])}
            if self.get_iou(bb2,bb1) > iou_limit:
                chosen_bboxes.append(bbox_to_compare)
        return chosen_bboxes

    #https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation
    def get_iou(self, bb1, bb2):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.

        Parameters
        ----------
        bb1 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x1, y1) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
        bb2 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x, y) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner

        Returns
        -------
        float
            in [0, 1]
        """
        assert bb1['x1'] < bb1['x2']
        assert bb1['y1'] < bb1['y2']
        assert bb2['x1'] < bb2['x2']
        assert bb2['y1'] < bb2['y2']

        # determine the coordinates of the intersection rectangle
        x_left = max(bb1['x1'], bb2['x1'])
        y_top = max(bb1['y1'], bb2['y1'])
        x_right = min(bb1['x2'], bb2['x2'])
        y_bottom = min(bb1['y2'], bb2['y2'])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # compute the area of both AABBs
        bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
        bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        assert iou >= 0.0
        assert iou <= 1.0
        return iou

    def resize(self, img, bboxes, new_w, new_h):
        w_ratio = new_w / img.shape[1]
        h_ratio = new_h / img.shape[0]
        img = cv2.resize(img, (new_w, new_h))
        for bbox in bboxes:
            bbox[0] = bbox[0] * w_ratio
            bbox[2] = bbox[2] * w_ratio
            bbox[1] = bbox[1] * h_ratio
            bbox[3] = bbox[3] * h_ratio
        return img, bboxes
