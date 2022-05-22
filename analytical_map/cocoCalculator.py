import json
from pycocotools.coco import COCO
import numpy as np
import os
import copy
from nptyping import NDArray
from typing import Tuple

from analytical_map.params import cocoParams
from analytical_map.tools.dump_json import dump_final_results_json as _dump_final_results_json
from analytical_map.tools.draw_chart import *


class COCOCalculator():
    def __init__(self, middle_file: str, result_dir: str, image_dir: str, params: cocoParams) -> None:
        """Init

        Args:
            middle_file (str): Path of a middle file
            result_dir (str): Path of a result directory
            image_dir (str): Path of an image directory
            params (cocoParams): Params for calculations
        """

        self.cocoGt = None
        self.cocoDt = None
        assert self.init_coco(middle_file)

        self.result_dir = result_dir
        self.image_dir = image_dir
        assert os.path.isdir(self.image_dir)
        # Default variables
        self.type = ['Match', 'LC', 'DC', 'Cls', 'Loc', 'Bkg', 'Miss']
        self.type_color = {'Match': (10, 20, 190), 'LC': (100, 20, 190), 'DC': (30, 140, 140),
                           'Cls': (190, 20, 100), 'Loc': (190, 30, 30), 'Bkg': (50, 50, 50), 'Miss': (80, 20, 170)}
        self.cats = self.cocoGt.loadCats(self.cocoGt.getCatIds())
        self.results = {'precision': [], 'recall': [], 'ap': []}
        self.area_all = [0, 10000000000.0]

        self.is_ap_calculated = False
        self.is_precision_calculated = False
        self.is_recall_calculated = False

        self.params = params
        self.params.area_rng = np.insert(
            params.area_rng, 0, self.area_all, axis=0)

    def init_coco(self, middle_file: str) -> bool:
        """Initialize cocoGt and cocoDt with a midddle file

        Args:
            middle_file (str): Path of the middle file

        Returns:
            bool: True if cocoGt and cocoDt are generated.
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

    def precision_calculate(self) -> None:
        """Calculate precision
        """

        cat_list = [id for id in self.cocoDt.getCatIds()]
        cat_list.append(self.cocoDt.getCatIds())
        category_names = [
            self.cats[cat-1]['name'] if not isinstance(cat, list) else 'single_category' for cat in cat_list]

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

            ratio = {k: round(v / num_dts, 3)
                     for k, v in counts.items()}
            self.results['precision'].append({
                'category': category_names[id_cat], 'score': precision, 'ratio': ratio})

            self.is_precision_calculated = True

    def recall_calculate(self) -> None:
        """Calculate recall
        """

        cat_list = [id for id in self.cocoGt.getCatIds()]
        cat_list.append(self.cocoGt.getCatIds())

        category_names = [
            self.cats[cat-1]['name'] if not isinstance(cat, list) else 'single_category' for cat in cat_list]

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
            ratio = {k: round(v / num_gts, 3)
                     for k, v in counts.items()}

            self.results['recall'].append({
                'category': category_names[id_cat], 'score': recall, 'ratio': ratio})

            self.is_recall_calculated = True

    def ap_calculate(self) -> None:
        """Calculate APs
        """

        def calc_ap(gts: dict, dts: dict, recall_inter: NDArray) -> Tuple[list, list, list, list]:
            """_summary_

            Args:
                gts (dict): _description_
                dts (dict): _description_
                recall_inter (NDArray): _description_

            Returns:
                Tuple[list, list, list, list]: score, precision, recall, precision_inter
            """
            num_gts = len(gts)

            inds = np.argsort([-d['score'] for d in dts], kind='mergesort')
            dts = [dts[i] for i in inds]

            x_score = np.zeros(len(dts))
            y_prec = np.zeros(len(dts))
            y_recall = np.zeros(len(dts))
            y_prec_inter = np.zeros(recall_inter.shape)

            if num_gts != 0:
                count_TP = 0
                for id, dt in enumerate(dts):
                    if dt['eval']['count'] == 'TP':
                        count_TP += 1
                    x_score[id] = dt['score']
                    y_prec[id] = count_TP / (id + 1)
                    y_recall[id] = count_TP / num_gts

                _y_prec = np.concatenate([[1], y_prec, [0]])
                _y_recall = np.concatenate([[0], y_recall, [1]])

                ids = np.searchsorted(_y_recall, recall_inter, side='left')
                for i, id in enumerate(ids):
                    y_prec_inter[i] = _y_prec[id]
                y_prec_inter = np.maximum.accumulate(y_prec_inter[::-1])[::-1]

            return x_score.tolist(), y_prec.tolist(), y_recall.tolist(), y_prec_inter.tolist()

        cat_list = [id for id in self.cocoGt.getCatIds()]
        cat_list.append(self.cocoGt.getCatIds())

        category_names = [
            self.cats[cat-1]['name'] if not isinstance(cat, list) else 'single_category' for cat in cat_list]
        area_names = ['area_' + str(area[0]) + '_' + str(area[1]) if not np.all(area == self.area_all)
                      else 'area_all' for area in self.params.area_rng]

        for id_cat, cat in enumerate(cat_list):

            for id_area, area in enumerate(self.params.area_rng):

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

                    score, prec_raw, recall_raw, prec_inter = calc_ap(
                        _gts, _dts, self.params.recall_inter)

                    # Anotehr way of calculating mAP.
                    # for i in range(1, len(prec_inter)):
                    #     ap[t] += prec_inter[i]*(self.params.recall_inter[i] -
                    #                               self.params.recall_inter[i-1])

                    # AP = 1/N sum(prec_inter)
                    ap[t] = np.average(prec_inter)

                ap_ratio = {k: round(v - ap['Match'], 3) if k != 'Match' else round(ap['Match'], 3)
                            for k, v in ap.items()}
                ap_ratio_normalized = {k: v/sum(ap_ratio.values()) if v != 0 else 0
                                       for k, v in ap_ratio.items()}

                self.results['ap'].append({
                    'category': category_names[id_cat], 'area': area_names[id_area], 'ap': round(ap['Match'], 3), 'ratio': ap_ratio_normalized,
                    'score': score, 'recall_raw': recall_raw, 'prec_raw': prec_raw, 'recall_inter': self.params.recall_inter.tolist(), 'prec_inter': prec_inter})

        self.is_ap_calculated = True

    def dump_final_results_json(self, final_file: str = 'final_results.json') -> None:
        """Dump final results

        Args:
            final_file (str, optional): Final result file's name. Defaults to 'final_results.json'.
        """
        _dump_final_results_json(
            self.cocoGt, self.params, self.results, self.result_dir, final_file)


if __name__ == '__main__':
    path_to_coco_dir = "example/data/"
    path_to_result_dir = "example/results/"
    path_to_middle_file = os.path.join(path_to_result_dir, 'middle_file.json')
    path_to_image_dir = os.path.join(path_to_coco_dir, 'images')

    p = cocoParams(recall_inter=np.arange(0, 1.01, 0.1), area_rng=np.array([
        [0, 1024], [1024, 9216], [9216, 10000000000.0]]))
    cocoCalc = COCOCalculator(path_to_middle_file,
                              path_to_result_dir, path_to_image_dir, p)
    cocoCalc.precision_calculate()
    cocoCalc.recall_calculate()
    cocoCalc.ap_calculate()
    cocoCalc.dump_final_results_json(final_file='final_results.json')
