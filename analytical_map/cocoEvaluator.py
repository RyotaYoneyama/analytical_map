from pycocotools.coco import COCO
import numpy as np
import os
from nptyping import NDArray
import copy
from analytical_map.params import COCOParams
from analytical_map.tools.dump_json import dump_middle_file_json as _dump_middle_file_json


class COCOEvaluator():
    def __init__(self, cocoGt_file: str, cocoDt_file: str, result_dir: str, params: COCOParams) -> None:
        """Init

        Args:
            cocoGt_file (str): COCO ground truth path
            cocoDt_file (str): COCO detection file path
            result_dir (str): Output path
            params (COCOParams): Parameters for evaluations
        """

        # Input
        self.cocoGt = None
        self.cocoDt = None
        self.cats = None
        assert self.init_coco(
            cocoGt_file, cocoDt_file)

        self.result_dir = result_dir

        # Fixed variables
        self.type = ['Match', 'LC', 'DC', 'Cls', 'Loc', 'Bkg', 'Miss']
        self.type_order = {'Match': 0, 'LC': 1, 'DC': 1,
                           'Cls': 2, 'Loc': 3, 'Bkg': 4, 'Miss': 4, None: 5}

        self.is_evaluated = False

        # User variables
        self.params = params

    def init_coco(self, cocoGt_file: str, cocoDt_file: str) -> bool:
        """Initialize coco data

        Args:
            cocoGt_file (str): COCO ground truth path
            cocoDt_file (str): COCO detection file path

        Returns:
            boo: True if cocoGt and cocoDt exist.
        """
        if cocoGt_file is not None and cocoDt_file is not None:
            if os.path.isfile(cocoGt_file) and os.path.isfile(cocoDt_file):
                cocoGt = COCO(cocoGt_file)
                _cocoDt = cocoGt.loadRes(cocoDt_file)
                dts_tmp = [dt for dt in _cocoDt.dataset['annotations']
                           if dt['score'] >= self.params.score_thresh]  # remove low scores
                cocoDt = cocoGt.loadRes(dts_tmp)

                eval_dict = {'eval': {'count': None,
                                      'type': None, 'corr_id': None, 'iou': None}}
                _gts = cocoGt.loadAnns(cocoGt.getAnnIds())
                _dts = cocoDt.loadAnns(cocoDt.getAnnIds())
                # print(_dts)
                _ = [g.update(eval_dict) for g in _gts]
                _ = [d.update(eval_dict) for d in _dts]
                self.cocoGt = cocoGt
                self.cocoDt = cocoDt
                self.cats = self.cocoGt.loadCats(self.cocoGt.getCatIds())
                return True
            else:
                print('ERROR:Could not read files')
                return False
        else:
            print('ERROR:Could not read files')
            return False

    def evaluate(self) -> None:
        """ Evaluate all images by repeating eval_per_img for all images.
        """
        if self.is_evaluated == False:
            img_ids = self.cocoGt.getImgIds()
            for img_id in img_ids:
                if self.eval_per_img(self.cocoGt, self.cocoDt, img_id,
                                     self.type_order, self.params.iou_thresh, self.params.iou_loc) == False:
                    self.is_evaluated = False
                    break
            self.is_evaluated = True
        else:
            print("Already evaluated")

    def eval_per_img(self, cocoGt: COCO, cocoDt: COCO, imgId: list, type_order: dict, iou_thresh: float, iou_loc: float) -> bool:
        """Evaluate bounding boxes in one image.

        Args:
            cocoGt (COCO): COCO ground truth instance
            cocoDt (COCO): COCO detection instance
            imgId (list): Image Id
            type_order (dict): type_order
            iou_thresh (float): Threshold for IoU
            iou_loc (float): Threshold for IoU to define 'Localization(Loc)' error.

        Returns:
            bool: True if eval_per_img is done correctly.
        """

        if type_order != {'Match': 0, 'LC': 1, 'DC': 1, 'Cls': 2, 'Loc': 3, 'Bkg': 4, 'Miss': 4, None: 5}:
            print('ERROR:Eval per image, type order', type_order)
            return False

        # Load all gts and dts in the image.
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

                # Calculate IoU of one gt and all dts in the image.
                # iou : np.array(NUM_DTS)
                iou = self.iou_per_single_gt(bb_gt, bb_dts)

                # Match boolean for all categories.
                bool_iou_all = iou >= iou_thresh
                # Loc boolean for all categories.
                bool_loc_all = np.logical_and(
                    iou >= iou_loc, iou < iou_thresh)

                # Category match boolean. bool_cat_all:np.array(NUM_DTS)
                cat_gt = np.array(gt['category_id'])
                cat_dts = np.array([dt['category_id'] for dt in dts])
                bool_cat_all = cat_dts == cat_gt

                # Matched ids
                Match_boolean = np.logical_and(
                    bool_cat_all, bool_iou_all)

                # loc ids
                Loc_boolean = np.logical_and(
                    bool_loc_all, bool_cat_all)

                # Category match ids
                Cat_boolean = np.logical_and(
                    bool_iou_all, np.logical_not(Match_boolean))

                # Match, DC, FC
                id_dets_match = np.where(Match_boolean == True)[0]
                for id_det in id_dets_match:
                    dt = dts[id_det]
                    # TP if gt is not assinged and dt is not TP-
                    if gt['eval']['type'] != "Match" and dt['eval']['type'] != "Match":
                        dt['eval'] = {"count": "TP", "type": "Match",
                                      "corr_id": gt['id'], 'iou': iou[id_det]}
                        gt['eval'] = {"count": "TP", "type": "Match",
                                      "corr_id": dt['id'], 'iou': iou[id_det]}
                        continue
                    # Double count if gt_assigned is assigned
                    elif gt['eval']['type'] == "Match" and dt['eval']['type'] != "Match":
                        dt['eval'] = {"count": "FP", "type": "DC",
                                      "corr_id": gt['id'], 'iou': iou[id_det]}
                    # Less count(LC) if all detections are already assigned
                    if id_det == id_dets_match[-1]:
                        if gt['eval']['type'] != "Match" and dt['eval']['type'] == "Match":
                            gt['eval'] = {
                                "count": "FN", "type": "LC", "corr_id": dts[id_dets_match[0]]['id'], 'iou': iou[id_det]}

                # Cls
                id_dets_cat = np.where(Cat_boolean == True)[0]
                for id_det in id_dets_cat:
                    dt = dts[id_det]
                    if type_order[dt['eval']['type']] > type_order['Cls']:
                        dt['eval'] = {"count": "FP", "type": "Cls",
                                      "corr_id": gt['id'], 'iou': iou[id_det]}
                    if type_order[gt['eval']['type']] > type_order['Cls']:
                        gt['eval'] = {"count": "FN", "type": "Cls",
                                      "corr_id": dt['id'], 'iou': iou[id_det]}

                # Loc
                id_dets_loc = np.where(Loc_boolean == True)[0]
                for id_det in id_dets_loc:
                    dt = dts[id_det]
                    if type_order[dt['eval']['type']] > type_order['Loc']:
                        dt['eval'] = {"count": "FP", "type": "Loc",
                                      "corr_id": gt['id'], 'iou': iou[id_det]}
                    if type_order[gt['eval']['type']] > type_order['Loc']:
                        gt['eval'] = {"count": "FN", "type": "Loc",
                                      "corr_id": dt['id'], 'iou': iou[id_det]}

                # No match
                if type_order[gt['eval']['type']] >= type_order[None]:
                    gt['eval'] = {"count": "FN", "type": "Miss",
                                  "corr_id": None, 'iou': None}

            # Bkg, if detections are not assigned yet
            for dt in dts:
                if type_order[dt['eval']['type']] >= type_order[None]:
                    dt['eval'] = {"count": "FP", "type": "Bkg",
                                  "corr_id": None, 'iou': None}

        else:
            # Miss if there are gts but no dts exist.
            if exist_gts and not exist_dts:
                for gt in gts:
                    gt['eval'] = {"count": "FN", "type": "Miss",
                                  "corr_id": None, 'iou': None}
            # Bkg, if dts exist but no gts exist.
            elif not exist_gts and exist_dts:
                for dt in dts:
                    dt['eval'] = {"count": "FP", "type": "Bkg",
                                  "corr_id": None, 'iou': None}
        return True

    def iou_per_single_gt(self, gt_bb: NDArray, dt_bbs: NDArray) -> NDArray:
        """Calculate IoU between one gt and multiple dts.

        Args:
            gt_bb (NDArray): 1x1 Bounding boxes of gts
            dt_bbs (NDArray): NUM_dts Bounding boxes of dts

        Returns:
            NDArray: IoU(Num_dts)
        """

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

    def dump_middle_file_json(self, middle_file: str = 'middle_file.json'):
        """Dump a middle file containing dts, gts with count and types.
        """
        _dump_middle_file_json(self.cocoGt, self.cocoDt,
                               self.params, self.result_dir, middle_file)


if __name__ == '__main__':
    path_to_coco_dir = "example/data/"
    path_to_result_dir = "example/results/"
    path_to_gt = os.path.join(path_to_coco_dir, 'coco', 'gt.json')
    path_to_dt = os.path.join(path_to_coco_dir, 'coco', 'dt.json')

    p = COCOParams(iou_thresh=0.5, iou_loc=0.2)
    cocoEval = COCOEvaluator(path_to_gt, path_to_dt,
                             path_to_result_dir, p)
    cocoEval.evaluate()
    cocoEval.dump_middle_file_json(middle_file='middle_file.json')
