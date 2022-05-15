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


class COCOevalDet(COCO):
    def __init__(self, result_dir, image_dir, cocoGt_file=None, cocoDt_file=None, eval_file=None):
        super().__init__()

        assert self.init_coco(cocoGt_file, cocoDt_file, eval_file)

        self.image_dir = image_dir
        self.result_dir = result_dir
        assert self.init_dirs()

        # Default variables
        self.type = ['Match', 'LC', 'DC', 'Cls', 'Loc', 'Bkg', 'Miss']
        self.type_order = {'Match': 0, 'LC': 1, 'DC': 1,
                           'Cls': 2, 'Loc': 3, 'Bkg': 4, 'Miss': 4, None: 5}
        self.type_color = {'Match': (10, 20, 190), 'LC': (100, 20, 190), 'DC': (30, 140, 140),
                           'Cls': (190, 20, 100), 'Loc': (190, 30, 30), 'Bkg': (50, 50, 50), 'Miss': (80, 20, 170)}
        self.cats = self.cocoGt.loadCats(self.cocoGt.getCatIds())
        self.results = {'precision': [], 'recall': [], 'ap': []}
        self.is_evaluated = False

        # User variables
        self.recall_inter = np.arange(0, 1.01, 0.1)
        self.iou_thresh = 0.5
        self.iou_loc = 0.2
        self.area_rng = [[0, 10000000000.0], [0, 1024],
                         [1024, 9216], [9216, 10000000000.0]]

    def init_dirs(self):

        if not os.path.isdir(self.image_dir):
            print('No image dir')
            return False

        os.makedirs(self.result_dir, exist_ok=True)

        return True

    def init_coco(self, cocoGt_file, cocoDt_file, eval_file):

        if cocoGt_file is not None and cocoDt_file is not None:
            if os.path.isfile(cocoGt_file) and os.path.isfile(cocoDt_file):
                self.cocoGt = COCO(cocoGt_file)
                self.cocoDt = self.cocoGt.loadRes(cocoDt_file)

                eval_dict = {'eval': {'count': None,
                                      'type': None, 'corr_id': None, 'iou': None}}
                _gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds())
                _dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds())
                _ = [g.update(eval_dict) for g in _gts]
                _ = [d.update(eval_dict) for d in _dts]
                return True

        elif eval_file is not None:
            if os.path.isfile(eval_file):
                self.cocoGt = COCO(eval_file)
                _cocoDt = json.load(open(eval_file))['detections']
                self.cocoDt = self.cocoGt.loadRes(_cocoDt)
                self.is_evaluated = True
            return True
        else:
            print('ERROR:Could not read files')
            return False

    def eval(self):
        if self.is_evaluated == False:
            img_ids = self.cocoGt.getImgIds()
            for img_id in img_ids:
                self.eval_per_img(img_id)
            self.is_evaluated = True
        else:
            print("Already evaluated")

    def eval_per_img(self, imgId):

        Id_gts = self.cocoGt.getAnnIds(imgIds=imgId, iscrowd=None)
        Id_dts = self.cocoDt.getAnnIds(imgIds=imgId, iscrowd=None)

        gts = self.cocoGt.loadAnns(Id_gts)
        dts = self.cocoDt.loadAnns(Id_dts)

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

                TP_all_cat_boolean = iou > self.iou_thresh
                TP_loc_all_cat_boolean = iou > self.iou_loc

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
                    if self.type_order[dt['eval']['type']] > self.type_order['Cls']:
                        dt['eval'] = {"count": "FP", "type": "Cls",
                                      "corr_id": gt['id'], 'iou': iou[id_det]}
                    if self.type_order[gt['eval']['type']] > self.type_order['Cls']:
                        gt['eval'] = {"count": "FN", "type": "Cls",
                                      "corr_id": dt['id'], 'iou': iou[id_det]}

                # Count location error
                id_dets_loc = np.where(TP_loc_boolean == True)[0]
                for id_det in id_dets_loc:
                    dt = dts[id_det]
                    if self.type_order[dt['eval']['type']] > self.type_order['Loc']:
                        dt['eval'] = {"count": "FP", "type": "Loc",
                                      "corr_id": gt['id'], 'iou': iou[id_det]}
                    if self.type_order[gt['eval']['type']] > self.type_order['Loc']:
                        gt['eval'] = {"count": "FN", "type": "Loc",
                                      "corr_id": dt['id'], 'iou': iou[id_det]}

                # No match
                if self.type_order[gt['eval']['type']] > self.type_order[None]:
                    gt['eval'] = {"count": "FN", "type": "Miss",
                                  "corr_id": None, 'iou': None}

            # Count Bkg, if detections are not assigned yet
            for dt in dts:
                if self.type_order[dt['eval']['type']] > self.type_order[None]:
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

    def iou_per_single_gt(self, gt_bb, dt_bbs):

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

    def precision_analyze(self):

        if self.eval == False:
            print('Evaluation should be done first.')
            return False
        cat_list = [id for id in self.cocoDt.getCatIds()]
        cat_list.append(self.cocoDt.getCatIds())
        category_names = [
            self.cats[cat-1]['name'] if not isinstance(cat, list) else 'all' for cat in cat_list]

        for id_cat, cat in enumerate(cat_list):

            dt_ids = self.cocoDt.getAnnIds(catIds=cat)
            dts = self.cocoDt.loadAnns(ids=dt_ids)
            counts = {'Match': 0, 'Loc': 0, 'DC': 0, 'LC': 0,
                      'Cls': 0, 'Bkg': 0, 'Miss': 0}
            num_dts = len(dts)

            for t in self.type:
                _count = len(
                    [dt for dt in dts if dt['eval']['type'] == t])
                counts[t] = _count

            precision = round(counts['Match'] / num_dts, 3)

            error_ratio = {}
            for k, v in counts.items():
                if k != 'Match':
                    error_ratio[k] = round(v / num_dts, 3)
            self.results['precision'].append({
                'category': category_names[id_cat], 'score': precision, 'error': error_ratio})

            dir_fig_precision = os.path.join(
                self.result_dir, 'figures', 'precision')
            os.makedirs(dir_fig_precision, exist_ok=True)

            _counts = np.array([[k, v]
                                for k, v in counts.items() if v != 0])
            counts_v = _counts[:, 1]
            counts_k = _counts[:, 0]

            fig_pie, ax_pie = plt.subplots()
            ax_pie.set_title(category_names[id_cat] + "_precision_ratio")
            ax_pie.pie(counts_v, labels=counts_k, autopct="%1.1f %%")
            fig_pie.savefig(os.path.join(dir_fig_precision,
                                         category_names[id_cat] + "_precision_ratio.png"))

        return True

    def recall_analyze(self):

        if self.eval == False:
            print('Evaluation should be done first.')
            return False

        cat_list = [id for id in self.cocoGt.getCatIds()]
        cat_list.append(self.cocoGt.getCatIds())

        category_names = [
            self.cats[cat-1]['name'] if not isinstance(cat, list) else 'all' for cat in cat_list]

        for id_cat, cat in enumerate(cat_list):

            gt_ids = self.cocoGt.getAnnIds(catIds=cat)
            gts = self.cocoGt.loadAnns(ids=gt_ids)
            counts = {'Match': 0, 'Loc': 0, 'DC': 0, 'LC': 0,
                      'Cls': 0, 'Bkg': 0, 'Miss': 0}
            num_gts = len(gts)

            for t in self.type:
                _count = len(
                    [gt for gt in gts if gt['eval']['type'] == t])
                counts[t] = _count

            recall = round(counts['Match'] / num_gts, 3)

            error_ratio = {}
            for k, v in counts.items():
                if k != 'Match':
                    error_ratio[k] = round(v / num_gts, 3)

            self.results['recall'].append({
                'category': category_names[id_cat], 'score': recall, 'error': error_ratio})

            dir_fig_recall = os.path.join(self.result_dir, 'figures', 'recall')
            os.makedirs(dir_fig_recall, exist_ok=True)

            _counts = np.array([[k, v]
                                for k, v in counts.items() if v != 0])
            counts_v = _counts[:, 1]
            counts_k = _counts[:, 0]
            fig_pie, ax_pie = plt.subplots()
            ax_pie.set_title(category_names[id_cat] + "_recall_ratio")
            ax_pie.pie(counts_v, labels=counts_k, autopct="%1.1f %%")
            fig_pie.savefig(os.path.join(
                dir_fig_recall, category_names[id_cat] + "_recall_ratio.png"))

        return True

    def ap_analyze(self):

        if self.eval == False:
            print('Evaluation should be done first.')
            return False

        def calc_ap(gts, dts, recall_inter):
            num_gts = len(gts)

            inds = np.argsort([-d['score'] for d in dts], kind='mergesort')
            dts = [dts[i] for i in inds]

            x_conf = np.zeros(len(dts))
            y_pres = np.zeros(len(dts))
            y_recall = np.zeros(len(dts))
            y_pres_inter = np.zeros(recall_inter.shape)

            if num_gts == 0:
                return x_conf, y_pres, y_recall, y_pres_inter

            count_TP = 0
            for id, dt in enumerate(dts):
                if dt['eval']['count'] == 'TP':
                    count_TP += 1
                x_conf[id] = dt['score']
                y_pres[id] = count_TP / (id + 1)
                y_recall[id] = count_TP / num_gts

            _y_pres = np.concatenate([[1], y_pres, [0]])
            _y_recall = np.concatenate([[0], y_recall, [1]])

            ids = np.searchsorted(_y_recall, recall_inter, side='left')
            for i, id in enumerate(ids):
                y_pres_inter[i] = _y_pres[id]
            y_pres_inter = np.maximum.accumulate(y_pres_inter[::-1])[::-1]

            return x_conf, y_pres, y_recall, y_pres_inter

        cat_list = [id for id in self.cocoGt.getCatIds()]
        cat_list.append(self.cocoGt.getCatIds())

        category_names = [
            self.cats[cat-1]['name'] if not isinstance(cat, list) else 'all' for cat in cat_list]
        area_names = ['area_' + str(area[0]) + '_' + str(area[1]) if area != [0, 10000000000.0]
                      else 'all' for id_area, area in enumerate(self.area_rng)]

        for id_cat, cat in enumerate(cat_list):

            for id_area, area in enumerate(self.area_rng):

                dt_ids = self.cocoDt.getAnnIds(catIds=cat, areaRng=area)
                gt_ids = self.cocoGt.getAnnIds(catIds=cat, areaRng=area)
                dts = self.cocoDt.loadAnns(ids=dt_ids)
                gts = self.cocoGt.loadAnns(ids=gt_ids)

                ap = dict.fromkeys(self.type, 0)

                for t in self.type:
                    _dts = copy.deepcopy(dts)
                    _gts = copy.deepcopy(gts)
                    if t != 'Match':
                        for gt in _gts:
                            if gt['eval']['type'] == t:
                                gt['eval']['count'] = 'TP'

                        for dt in _dts:
                            if dt['eval']['type'] == t:
                                dt['eval']['count'] = 'TP'

                    x_conf, y_pres, y_recall, y_pres_inter = calc_ap(
                        _gts, _dts, self.recall_inter)

                    for i in range(1, len(y_pres_inter)):
                        ap[t] += y_pres_inter[i]*(self.recall_inter[i] -
                                                  self.recall_inter[i-1])

                error_ratio = {}
                for k in self.type:
                    if k != 'Match':
                        error_ratio[k] = round(ap[k] - ap['Match'], 3)

                self.results['ap'].append({
                    'category': category_names[id_cat], 'area': area_names[id_area], 'score': round(ap['Match'], 3), 'error': error_ratio})

                dir_fig_ap = os.path.join(
                    self.result_dir, 'figures', 'ap', category_names[id_cat], area_names[id_area])
                os.makedirs(dir_fig_ap, exist_ok=True)

                fig_conf, ax_conf = plt.subplots()
                ax_conf.plot(x_conf, y_pres, color='red')
                ax_conf.plot(x_conf, y_recall, color='green')
                ax_conf.set_xlim(-0.05, 1.05)
                ax_conf.set_ylim(-0.05, 1.05)
                ax_conf.legend(["Precision", "Recall"])
                ax_conf.set_xlabel('confidence')
                fig_conf.savefig(os.path.join(dir_fig_ap,  "pr_conf.png"))

                fig_pr, ax_pr = plt.subplots()

                ax_pr.plot(y_recall, y_pres, marker='.', color='royalblue')
                ax_pr.plot(self.recall_inter, y_pres_inter,
                           marker='x', linewidth=0, color='orange')
                ax_pr.set_xlabel('Precision')
                ax_pr.set_xlabel('Recall')
                ax_pr.legend(["PR_raw", "PR_inter"])
                ax_pr.set_xlim(-0.05, 1.05)
                ax_pr.set_ylim(-0.05, 1.05)
                fig_pr.savefig(os.path.join(dir_fig_ap,  "pr_curve.png"))

            mAP = np.round(sum([ap['score'] for ap in self.results['ap']
                                if ap['category'] != 'all' and ap['area'] == 'all'])/len(self.cats), 3)

        self.results['ap'].append({
            'map': mAP})
        return True

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

    def dump_middle_results_json(self):

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
        fw = open(os.path.join(self.result_dir, 'middle_results.json'), 'w')
        json.dump(js, fw, indent=2)

    def dump_final_results_json(self):

        fw = open(os.path.join(self.result_dir, 'final_results.json'), 'w')
        json.dump([self.results], fw, indent=2)


if __name__ == '__main__':
    path_to_coco_dir = "/home/ryota/dl/tools/analytical_map/dog/"
    path_to_gt = os.path.join(path_to_coco_dir, 'coco', 'gt.json')
    path_to_dt = os.path.join(path_to_coco_dir, 'coco', 'dt.json')
    path_to_image_dir = os.path.join(path_to_coco_dir, 'images')
    path_to_result_dir = os.path.join(path_to_coco_dir, 'results')

    cocoEvalDet = COCOevalDet(
        path_to_result_dir, path_to_image_dir, path_to_gt, path_to_dt)
    cocoEvalDet.eval()
    cocoEvalDet.precision_analyze()
    cocoEvalDet.recall_analyze()
    cocoEvalDet.ap_analyze()
    cocoEvalDet.visualize()
    cocoEvalDet.dump_middle_results_json()
    cocoEvalDet.dump_final_results_json()
