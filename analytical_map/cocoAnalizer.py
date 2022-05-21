import time
import json
import collections as cl
from pycocotools.coco import COCO
import numpy as np
import os
import sys
import io
import matplotlib.pyplot as plt
import copy
import cv2
from typing import Tuple
from dataclasses import asdict
import copy

from analytical_map.params import cocoParams


class COCOAnalizer(COCO):
    def __init__(self, middle_file, result_dir, params: cocoParams):
        super().__init__()

        self.cocoGt = None
        self.cocoDt = None
        assert self.init_coco(middle_file)

        self.result_dir = result_dir
        os.makedirs(self.result_dir, exist_ok=True)

        # Default variables
        self.type = ['Match', 'LC', 'DC', 'Cls', 'Loc', 'Bkg', 'Miss']
        self.cats = self.cocoGt.loadCats(self.cocoGt.getCatIds())
        self.results = {'precision': [], 'recall': [], 'ap': []}
        self.area_all = [0, 10000000000.0]

        # User variables
        self.params = params
        self.recall_inter = params.recall_inter
        self.area_rng = np.insert(params.area_rng, 0, self.area_all, axis=0)

    def init_coco(self, middle_file: str) -> bool:
        """_summary_

        Args:
            cocoGt_file (str): _description_
            cocoDt_file (str): _description_
            middle_file (str): _description_

        Returns:
            bool: _description_
        """
        if middle_file is not None:
            if os.path.isfile(middle_file):
                cocoGt = COCO(middle_file)
                _cocoDt = json.load(open(middle_file))['detections']
                cocoDt = cocoGt.loadRes(_cocoDt)
                if self.evaluation_check(cocoGt, cocoDt) == False:
                    return False

                self.cocoGt = cocoGt
                self.cocoDt = cocoDt
                return True
        else:
            print('ERROR:Could not read files')
            return False

    def evaluation_check(self, cocoGt: COCO, cocoDT: COCO) -> bool:
        """Check whether middle file is evaluated. 

        Args:
            cocoGt (COCO):  Ground truth
            cocoDT (COCO):  Detections

        Returns:
            bool: True if both cocoGt and cocoDT were evaluated.
        """
        gts = cocoGt.loadAnns(cocoGt.getAnnIds())
        dts = cocoDT.loadAnns(cocoDT.getAnnIds())
        is_evaluated_gts = all([True if 'eval' in gt.keys()
                                else False for gt in gts])
        is_evaluated_dts = all([True if 'eval' in dt.keys()
                                else False for dt in dts])
        return is_evaluated_gts and is_evaluated_dts

    def precision_analyze(self):

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

            error_ratio = {k: round(v / num_dts, 3)
                           for k, v in counts.items() if v != 'Match'}
            self.results['precision'].append({
                'category': category_names[id_cat], 'score': precision, 'error': error_ratio})

            dir_fig_precision = os.path.join(
                self.result_dir, 'figures', 'precision')
            os.makedirs(dir_fig_precision, exist_ok=True)

            _counts = {k: v for k, v in counts.items() if v != 0}

            fig_pie, ax_pie = plt.subplots()
            ax_pie.set_title(category_names[id_cat] + "_precision_ratio")
            ax_pie.pie(_counts.values(), labels=_counts.keys(),
                       autopct="%1.1f %%")
            fig_pie.savefig(os.path.join(dir_fig_precision,
                                         category_names[id_cat] + "_precision_ratio.png"))

    def recall_analyze(self):

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
            error_ratio = {k: round(v / num_gts, 3)
                           for k, v in counts.items() if v != 'Match'}

            self.results['recall'].append({
                'category': category_names[id_cat], 'score': recall, 'error': error_ratio})

            dir_fig_recall = os.path.join(self.result_dir, 'figures', 'recall')
            os.makedirs(dir_fig_recall, exist_ok=True)

            _counts = {k: v for k, v in counts.items() if v != 0}

            fig_pie, ax_pie = plt.subplots()
            ax_pie.set_title(category_names[id_cat] + "_recall_ratio")
            ax_pie.pie(_counts.values(), labels=_counts.keys(),
                       autopct="%1.1f %%")
            fig_pie.savefig(os.path.join(
                dir_fig_recall, category_names[id_cat] + "_recall_ratio.png"))

    def ap_analyze(self):

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
        area_names = ['area_' + str(area[0]) + '_' + str(area[1]) if not np.all(area == self.area_all)
                      else 'area_all' for area in self.area_rng]

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

                    # for i in range(1, len(y_pres_inter)):
                    #     ap[t] += y_pres_inter[i]*(self.recall_inter[i] -
                    #                               self.recall_inter[i-1])
                    ap[t] = np.average(y_pres_inter)
                ap_ratio = {k: v - ap['Match'] if k != 'Match' else ap['Match']
                            for k, v in ap.items()}
                error_ratio = {k: v for k, v in ap_ratio.items()
                               if k != 'Match'}
                ap_ratio_normalized = {k: v/sum(ap_ratio.values())
                                       for k, v in ap_ratio.items() if v != 0}

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

                fig_pie, ax_pie = plt.subplots()
                ax_pie.set_title("ap_ratio")
                ax_pie.pie(ap_ratio_normalized.values(),
                           labels=ap_ratio_normalized.keys(), autopct="%1.1f %%")
                fig_pie.savefig(os.path.join(dir_fig_ap, "ap_ratio.png"))

    def dump_final_results_json(self):
        def categories(cocoGt):
            return cocoGt.loadCats(cocoGt.getCatIds())

        def param2dict(params):
            _params = copy.deepcopy(params)
            _params.recall_inter = _params.recall_inter.tolist()
            _params.area_rng = _params.area_rng.tolist()
            tmp = [asdict(_params)]
            return tmp

        query_list = ["licenses", "info", "categories", "params",
                      "results"]
        js = cl.OrderedDict()
        for i in range(len(query_list)):
            tmp = ""
            if query_list[i] == "categories":
                tmp = categories(self.cocoGt)
            if query_list[i] == "params":
                tmp = param2dict(self.params)
            if query_list[i] == "results":
                tmp = [self.results]

            js[query_list[i]] = tmp

        fw = open(os.path.join(self.result_dir, 'final_results.json'), 'w')
        json.dump(js, fw, indent=2)


if __name__ == '__main__':
    path_to_coco_dir = "sample_data/"
    path_to_result_dir = "sample_results/"
    path_to_middle_file = os.path.join(path_to_result_dir, 'middle_file.json')

    p = cocoParams(recall_inter=np.arange(0, 1.01, 0.1), area_rng=np.array([
                   [0, 1024], [1024, 9216], [9216, 10000000000.0]]))
    cocoAnal = COCOAnalizer(path_to_middle_file, path_to_result_dir, p)
    cocoAnal.precision_analyze()
    cocoAnal.recall_analyze()
    cocoAnal.ap_analyze()
    cocoAnal.dump_final_results_json()
