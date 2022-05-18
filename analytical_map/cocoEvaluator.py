import time
import json
import collections as cl
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import os
import sys
import io
import matplotlib.pyplot as plt
import copy
import cv2
from typing import Tuple


class COCOEvaluator(COCO):
    def __init__(self, cocoGt_file, cocoDt_file, result_dir, image_dir):
        super().__init__()

        # Input
        self.cocoGt, self.cocoDt = self.init_coco(
            cocoGt_file, cocoDt_file)

        self.image_dir = image_dir
        self.result_dir = result_dir
        assert self.init_dirs(self.image_dir, self.result_dir)

        # Fixed variables
        self.type = ['Match', 'LC', 'DC', 'Cls', 'Loc', 'Bkg', 'Miss']
        self.type_order = {'Match': 0, 'LC': 1, 'DC': 1,
                           'Cls': 2, 'Loc': 3, 'Bkg': 4, 'Miss': 4, None: 5}
        self.cats = self.cocoGt.loadCats(self.cocoGt.getCatIds())
        self.is_evaluated = False

        # User variables
        self.iou_thresh = 0.5
        self.iou_loc = 0.2
        self.type_color = {'Match': (10, 20, 190), 'LC': (100, 20, 190), 'DC': (30, 140, 140),
                           'Cls': (190, 20, 100), 'Loc': (190, 30, 30), 'Bkg': (50, 50, 50), 'Miss': (80, 20, 170)}

    def init_dirs(self, image_dir: str, result_dir: str) -> bool:
        """Initialize directories

        Args:
            image_dir (str): Image directory path
            result_dir (str): Results(outputs) directory path

        Returns:
            bool: True if image_dir exists and result_dir is successfully crated. 
        """

        if not os.path.isdir(image_dir):
            print('No image dir')
            return False

        os.makedirs(result_dir, exist_ok=True)

        return True

    def init_coco(self, cocoGt_file: str, cocoDt_file: str) -> Tuple[COCO, COCO]:
        """_summary_

        Args:
            cocoGt_file (str): _description_
            cocoDt_file (str): _description_

        Returns:
            COCO, COCO: _description_
        """
        if cocoGt_file is not None and cocoDt_file is not None:
            if os.path.isfile(cocoGt_file) and os.path.isfile(cocoDt_file):
                cocoGt = COCO(cocoGt_file)
                cocoDt = cocoGt.loadRes(cocoDt_file)

                eval_dict = {'eval': {'count': None,
                                      'type': None, 'corr_id': None, 'iou': None}}
                _gts = cocoGt.loadAnns(cocoGt.getAnnIds())
                _dts = cocoDt.loadAnns(cocoDt.getAnnIds())
                _ = [g.update(eval_dict) for g in _gts]
                _ = [d.update(eval_dict) for d in _dts]
                return cocoGt, cocoDt
        else:
            print('ERROR:Could not read files')
            return False

    def eval(self):
        if self.is_evaluated == False:
            img_ids = self.cocoGt.getImgIds()
            for img_id in img_ids:
                if self.eval_per_img(self.cocoGt, self.cocoDt, img_id,
                                     self.type_order, self.iou_thresh, self.iou_loc) == False:
                    self.is_evaluated = False
                    break
            self.is_evaluated = True
        else:
            print("Already evaluated")

    def eval_per_img(self, cocoGt: COCO, cocoDt: COCO, imgId: list, type_order: dict, iou_thresh: float, iou_loc: float):

        if type_order != {'Match': 0, 'LC': 1, 'DC': 1, 'Cls': 2, 'Loc': 3, 'Bkg': 4, 'Miss': 4, None: 5}:
            print('ERROR:Eval per image, type order', type_order)
            return False

        Id_gts = cocoGt.getAnnIds(imgIds=imgId, iscrowd=None)
        Id_dts = cocoDt.getAnnIds(imgIds=imgId, iscrowd=None)

        gts = cocoGt.loadAnns(Id_gts)
        dts = cocoDt.loadAnns(Id_dts)

        # Sort detections by score
        inds = np.argsort([-d['score'] for d in dts], kind='mergesort')
        dts = [dts[i] for i in inds]

        exist_gts = bool(gts)
        exist_dts = bool(dts)

        if exist_gts and exist_dts:

            for gt in gts:

                bb_gt = np.array(gt["bbox"])
                bb_dts = np.array([dt["bbox"] for dt in dts])

                iou = self.iou_per_single_gt(bb_gt, bb_dts)

                TP_all_cat_boolean = iou > iou_thresh
                TP_loc_all_cat_boolean = iou > iou_loc

                cat_gt = np.array(gt['category_id'])
                cat_dts = np.array([dt['category_id'] for dt in dts])
                cat_match_boolean = cat_dts == cat_gt

                TP_boolean = np.logical_and(
                    cat_match_boolean, TP_all_cat_boolean)
                TP_loc_boolean = np.logical_and(
                    cat_match_boolean, TP_loc_all_cat_boolean)
                TP_cat_boolean = np.logical_and(
                    TP_all_cat_boolean, np.logical_not(TP_boolean))

                # Count TP, double count and  less counts(LC)
                id_dets_match = np.where(TP_boolean == True)[0]
                for id_det in id_dets_match:
                    dt = dts[id_det]
                    # TP if gt is not assinged and dt is not TP-
                    if gt['eval']['count'] != "TP" and dt['eval']['count'] != "TP":
                        dt['eval'] = {"count": "TP", "type": "Match",
                                      "corr_id": gt['id'], 'iou': iou[id_det]}
                        gt['eval'] = {"count": "TP", "type": "Match",
                                      "corr_id": dt['id'], 'iou': iou[id_det]}
                        continue
                    # Double count if gt_assigned is assigned
                    elif gt['eval']['count'] == "TP" and dt['eval']['count'] != "TP":
                        dt['eval'] = {"count": "FP", "type": "DC",
                                      "corr_id": gt['id'], 'iou': iou[id_det]}
                    # Less count(LC) if all detections are already assigned
                    if id_det == id_dets_match[-1]:
                        if gt['eval']['count'] != "TP" and dt['eval']['count'] == "TP":
                            gt['eval'] = {
                                "count": "FN", "type": "LC", "corr_id": dts[id_dets_match[0]]['id'], 'iou': iou[id_det]}

                # Count catgory mistakes
                id_dets_all = np.where(TP_cat_boolean == True)[0]
                for id_det in id_dets_all:
                    dt = dts[id_det]
                    if type_order[dt['eval']['type']] > type_order['Cls']:
                        dt['eval'] = {"count": "FP", "type": "Cls",
                                      "corr_id": gt['id'], 'iou': iou[id_det]}
                    if type_order[gt['eval']['type']] > type_order['Cls']:
                        gt['eval'] = {"count": "FN", "type": "Cls",
                                      "corr_id": dt['id'], 'iou': iou[id_det]}

                # Count location error
                id_dets_loc = np.where(TP_loc_boolean == True)[0]
                for id_det in id_dets_loc:
                    dt = dts[id_det]
                    if type_order[dt['eval']['type']] > type_order['Loc']:
                        dt['eval'] = {"count": "FP", "type": "Loc",
                                      "corr_id": gt['id'], 'iou': iou[id_det]}
                    if type_order[gt['eval']['type']] > type_order['Loc']:
                        gt['eval'] = {"count": "FN", "type": "Loc",
                                      "corr_id": dt['id'], 'iou': iou[id_det]}

                # No match
                if type_order[gt['eval']['type']] > type_order[None]:
                    gt['eval'] = {"count": "FN", "type": "Miss",
                                  "corr_id": None, 'iou': None}

            # Count Bkg, if detections are not assigned yet
            for dt in dts:
                if type_order[dt['eval']['type']] > type_order[None]:
                    dt['eval'] = {"count": "FP", "type": "Bkg",
                                  "corr_id": None, 'iou': None}

        else:
            if exist_gts and not exist_dts:
                for gt in gts:
                    gt['eval'] = {"count": "FN", "type": "Miss",
                                  "corr_id": None, 'iou': None}

            elif not exist_gts and exist_dts:
                for dt in dts:
                    dt['eval'] = {"count": "FP", "type": "Bkg",
                                  "corr_id": None, 'iou': None}
        return True

    def iou_per_single_gt(self, gt_bb: np.array, dt_bbs: np.array) -> np.array:

        gt_area = (gt_bb[2] + 1) \
            * (gt_bb[3] + 1)

        dt_areas = (dt_bbs[:, 2] + 1) \
            * (dt_bbs[:, 3] + 1)

        abx_min = np.maximum(gt_bb[0], dt_bbs[:, 0])  # xmin
        aby_min = np.maximum(gt_bb[1], dt_bbs[:, 1])  # ymin
        abx_max = np.minimum(gt_bb[0] + gt_bb[2],
                             dt_bbs[:, 0] + dt_bbs[:, 2])  # xmax
        aby_max = np.minimum(gt_bb[1] + gt_bb[3],
                             dt_bbs[:, 1] + dt_bbs[:, 3])  # ymax

        w = np.maximum(0, abx_max - abx_min + 1)
        h = np.maximum(0, aby_max - aby_min + 1)
        intersect = w*h

        iou = intersect / (gt_area + dt_areas - intersect)
        return iou

    def visualize(self):
        if self.eval == False:
            print('Evaluation should be done first.')
            return False

        dir_TP = os.path.join(self.result_dir, 'visualize', 'TP')
        dir_not_TP = os.path.join(self.result_dir, 'visualize', 'not_TP')
        os.makedirs(dir_TP, exist_ok=True)
        os.makedirs(dir_not_TP, exist_ok=True)

        img_ids = self.cocoGt.getImgIds()
        for img_id in img_ids:
            img = self.cocoGt.loadImgs(ids=img_id)[0]

            img_cv2 = cv2.imread(os.path.join(
                self.image_dir, img["file_name"]))
            gts_ids = self.cocoGt.getAnnIds(imgIds=img['id'],  iscrowd=None)
            dts_ids = self.cocoDt.getAnnIds(imgIds=img['id'],  iscrowd=None)

            gts_per_img = self.cocoGt.loadAnns(gts_ids)
            dts_per_img = self.cocoDt.loadAnns(dts_ids)

            is_all_TPs = True
            for gt in gts_per_img:

                if gt['eval']['count'] != 'TP':
                    is_all_TPs = False

                x_min = int(gt["bbox"][0])
                y_min = int(gt["bbox"][1])
                x_max = int(gt["bbox"][0]) + int(gt["bbox"][2])
                y_max = int(gt["bbox"][1]) + int(gt["bbox"][3])

                cv2.rectangle(img_cv2, (x_min, y_min),
                              (x_max, y_max), self.type_color[gt['eval']['type']], thickness=2)
                cv2.putText(img_cv2, str(gt['eval']['type']),
                            org=(x_min, y_min-5),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            color=self.type_color[gt['eval']['type']],
                            thickness=2,
                            lineType=cv2.LINE_4)

            for dt in dts_per_img:
                if dt['eval']['count'] != 'TP':
                    is_all_TPs = False

                x_min = int(dt["bbox"][0])
                y_min = int(dt["bbox"][1])
                x_max = int(dt["bbox"][0]) + int(dt["bbox"][2])
                y_max = int(dt["bbox"][1]) + int(dt["bbox"][3])

                cv2.rectangle(img_cv2, (x_min, y_min), (x_max, y_max), tuple(
                    [1.3*c for c in self.type_color[dt['eval']['type']]]))
                cv2.putText(img_cv2,  str(dt['eval']['type']),
                            org=(x_min, y_min-5),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            color=self.type_color[dt['eval']['type']],
                            thickness=1,
                            lineType=cv2.LINE_4)

            save_dir = dir_TP if is_all_TPs else dir_not_TP
            cv2.imwrite(os.path.join(save_dir, img["file_name"]), img_cv2)

        return True

    def dump_middle_file_json(self):

        def images(cocoGt):
            img_ids = cocoGt.getImgIds()
            return cocoGt.loadImgs(ids=img_ids)

        def categories(cocoGt):
            return cocoGt.loadCats(cocoGt.getCatIds())

        def annotations(cocoGt):
            annIds = cocoGt.getAnnIds()
            return cocoGt.loadAnns(ids=annIds)

        def detections(cocoDt):
            annIds = cocoDt.getAnnIds()
            return cocoDt.loadAnns(ids=annIds)

        query_list = ["licenses", "info", "categories", "images",
                      "annotations", "detections", "segment_info"]
        js = cl.OrderedDict()
        for i in range(len(query_list)):
            tmp = ""
            if query_list[i] == "categories":
                tmp = categories(self.cocoGt)
            if query_list[i] == "images":
                tmp = images(self.cocoGt)
            if query_list[i] == "annotations":
                tmp = annotations(self.cocoGt)
            if query_list[i] == "detections":
                tmp = detections(self.cocoDt)

            # save it
            js[query_list[i]] = tmp
        # write
        middle_file_path = os.path.join(self.result_dir, 'middle_file.json')
        fw = open(middle_file_path, 'w')
        json.dump(js, fw, indent=2)
        return middle_file_path


if __name__ == '__main__':
    path_to_coco_dir = "../sample_data/"
    path_to_result_dir = "../sample_results/"
    path_to_gt = os.path.join(path_to_coco_dir, 'coco', 'gt.json')
    path_to_dt = os.path.join(path_to_coco_dir, 'coco', 'dt.json')
    path_to_image_dir = os.path.join(path_to_coco_dir, 'images')

    cocoEval = COCOEvaluator(path_to_gt, path_to_dt,
                             path_to_result_dir, path_to_image_dir)
    cocoEval.eval()
    cocoEval.visualize()
    middle_file_path = cocoEval.dump_middle_file_json()
