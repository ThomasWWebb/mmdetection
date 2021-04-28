import json
import time
from typing import Union, List
import numpy as np
import os

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval, Params

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from matplotlib import pyplot as plt


class DetectionPerformanceEvaluation:

    def __init__(self, gt: Union[str, COCO], prediction: Union[List, str], params=None, th=0.5):
        if isinstance(gt, str):
            gt = COCO(gt)

        prediction_coco = dict()
        if isinstance(prediction, str):
            print('loading detectron output annotations into memory...')
            tic = time.time()
            prediction = json.load(open(prediction, 'r'))  # Loading the json file as an array of dicts
            assert type(prediction) == list, 'annotation file format {} not supported'.format(
                type(prediction))
            print('Done (t={:0.2f}s)'.format(time.time() - tic))

        for i, p in enumerate(prediction):
            p['id'] = i
            p['segmentation'] = []
            p['area'] = p['bbox'][2] * p['bbox'][3]
        # Adding these lines I give the detection file the xray format
        prediction_coco["annotations"] = prediction
        prediction_coco["images"] = gt.dataset["images"]
        prediction_coco["categories"] = gt.dataset["categories"]

        # COCO object instantiation
        prediction = COCO()
        prediction.dataset = prediction_coco
        prediction.createIndex()

        self.ground_truth = gt
        self.prediction = prediction
        self.eval = COCOeval(gt, prediction, iouType='bbox')
        self.params = self.eval.params
        self._imgIds = gt.getImgIds()
        self._catIds = gt.getCatIds()
        self.th = th
        if params:
            self.params = params
            self.eval.params = params
            self.eval.params.imgIds = sorted(self._imgIds)
            self.eval.params.catIds = sorted(self._catIds)

    def _build_no_cat_params(self):
        params = Params(iouType='bbox')
        params.maxDets = [100]
        params.areaRng = [[0 ** 2, 1e5 ** 2]]
        params.areaRngLbl = ['all']
        params.useCats = 0
        params.iouThrs = [self.th]
        return params

    def build_confussion_matrix(self, out_image_filename=None):
        params = self._build_no_cat_params()
        self.eval.params = params
        self.eval.params.imgIds = sorted(self._imgIds)
        self.eval.params.catIds = sorted(self._catIds)
        self.eval.evaluate()

        ann_true = []
        ann_pred = []

        for evalImg, ((k, _), ious) in zip(self.eval.evalImgs, self.eval.ious.items()):
            ann_true += evalImg['gtIds']
            if len(ious) > 0:
                valid_ious = (ious >= self.th) * ious
                matches = valid_ious.argmax(0)
                matches[valid_ious.max(0) == 0] = -1
                ann_pred += [evalImg['dtIds'][match] if match > -1 else -1 for match in matches]
            else:
                ann_pred += ([-1] * len(evalImg['gtIds']))

        y_true = [ann['category_id'] for ann in self.ground_truth.loadAnns(ann_true)]
        y_pred = [-1 if ann == -1 else self.prediction.loadAnns(ann)[0]['category_id'] for ann in ann_pred]
        y_true = [y + 1 for y in y_true]
        y_pred = [y + 1 for y in y_pred]
        cats = ['background'] + [cat['name'] for _, cat in self.ground_truth.cats.items()]
        cnf_mtx = confusion_matrix(y_true, y_pred, normalize='true')
        #print(cnf_mtx)


        ####TPR/FPR/TNR
        FP = cnf_mtx.sum(axis=0) - np.diag(cnf_mtx)  
        FN = cnf_mtx.sum(axis=1) - np.diag(cnf_mtx)
        TP = np.diag(cnf_mtx)
        TN = cnf_mtx.sum() - (FP + FN + TP)

        FP = FP.astype(float)
        FN = FN.astype(float)
        TP = TP.astype(float)
        TN = TN.astype(float)

        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP/(TP+FN)
        # Specificity or true negative rate
        TNR = TN/(TN+FP) 
        # Precision or positive predictive value
        PPV = TP/(TP+FP)
        # Negative predictive value
        NPV = TN/(TN+FN)
        # Fall out or false positive rate
        FPR = FP/(FP+TN)
        # False negative rate
        FNR = FN/(TP+FN)
        # False discovery rate
        FDR = FP/(TP+FP)
        # Overall accuracy
        ACC = (TP+TN)/(TP+FP+FN+TN)

        #Avg
        TPRav = sum(TPR[1:])/(len(TPR)-1)
        TNRav = sum(TNR[1:])/(len(TNR)-1)
        FPRav = sum(FPR[1:])/(len(FPR)-1)
        FNRav = sum(FNR[1:])/(len(FNR)-1)
        ACCav = sum(ACC[1:])/(len(ACC)-1)

        # print('***********************************************************************')
        # print('Stats from confusion Matrix:')
        # print('Class-wise:')
        # print(f'TPR: {TPR}\nTNR: {TNR}\nFPR: {FPR}\nFNR: {FNR}\nACC: {ACC}')
        # print('Overall:')
        # print(f'TPR: {round(TPRav,3)}\nTNR: {round(TNRav,3)}\nFPR: {round(FPRav,3)}\nFNR: {round(FNRav,3)}\nACC: {round(ACCav,3)}')
        # print('***********************************************************************')    
        ####

        cnf_mtx_display = ConfusionMatrixDisplay(cnf_mtx, cats)
        _, ax = plt.subplots(figsize=(10, 9))
        plt.rcParams.update({'font.size': 18})
        cnf_mtx_display.plot(ax=ax, values_format='.3f',xticks_rotation=45)
        if out_image_filename is not None:
            cnf_mtx_display.figure_.savefig(out_image_filename)
        #print(classification_report(y_true, y_pred, target_names=cats))
        pass

    def run_coco_metrics(self):
        self.eval.params = self.params
        self.eval.params.imgIds = sorted(self._imgIds)
        self.eval.params.catIds = sorted(self._catIds)
        self.eval.evaluate()
        self.eval.accumulate()
        self.eval.summarize()

        print('\n\n|====Class wise coco evaluation====|')
        for c_id in self._catIds:
            print(f'\n|____Category id: {c_id}')
            self.eval.params.catIds = [c_id]
            self.eval.evaluate()
            self.eval.accumulate()
            self.eval.summarize()


def build_params():
    params = Params(iouType='bbox')
    params.maxDets = [1, 100, 100]
    params.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
    params.areaRngLbl = ['all', 'small', 'medium', 'large']
    params.useCats = 1
    return params


def main():
    conf = 0.05
    iou = 0.5
    db = 'deei6'
    model = 'cascade_rcnn'
    #quality = 95
    backbone = 'r50' 
    for epoch in range(10, 40, 10):
        os.makedirs(f'../Results/statistics/{model}_{backbone}_e30_{db}_originalResolution_merged_raw/', exist_ok=True)
        if db == "deei6":
            gt_path = f'../datasets/{db}/annotations/deei6_bayer(zhl)_test.json'
        else:
            gt_path = f'../datasets/{db}/annotations/instances_test2017.json'
        # pred_path = f'statistics/{db}_{arch}/test_{conf}conf_{iou}iou.json'
        pred_path = f'../Results/{model}_{backbone}_e30_{db}_originalResolution_merged_raw/test_results_{epoch}.bbox.json'
        output_image = f'../Results/{model}_{backbone}_e30_{db}_originalResolution_merged_raw/confmat_bbox_{db}_e{epoch}.png'
        # output_image = f'statistics/{db}_{arch}/confmat_{conf}conf_{iou}iou.png'
        confusion_matrix_iou_threshold = 0.5

        params = build_params()  # Params for COCO metrics
        performance_evaluation = DetectionPerformanceEvaluation(gt_path, pred_path, params=params,
                                                                th=confusion_matrix_iou_threshold)
        performance_evaluation.build_confussion_matrix(output_image)
        performance_evaluation.run_coco_metrics()


if __name__ == '__main__':
    main()