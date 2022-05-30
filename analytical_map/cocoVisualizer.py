import json
from pycocotools.coco import COCO
import os
import cv2
from bokeh.io import save, output_file, export_png
from bokeh.plotting import figure, show
from bokeh.layouts import gridplot
import pandas as pd
import seaborn as sns

from analytical_map.params import COCOParams
from analytical_map.tools.dump_json import dump_final_results_json as _dump_final_results_json
from analytical_map.tools.draw_chart import *


class COCOVisualizer():
    def __init__(self, middle_file: str, results_file: str, result_dir: str, image_dir: str) -> None:
        """Init

        Args:
            middle_file (str): Path of a middle file.
            results_file (str): Path of a result file
            result_dir (str): Path of a result directory
            image_dir (str): Path of an image directory
        """

        self.cocoGt = None
        self.cocoDt = None
        self.is_precision_calculated = False
        self.is_recall_calculated = False
        self.is_ap_calculated = False
        self.is_evaluated = False

        assert self.read_middle_file(middle_file)
        assert self.read_result_file(results_file)

        self.result_dir = result_dir
        self.image_dir = image_dir
        assert os.path.isdir(self.image_dir)

        # Default variables
        self.type = ['Match', 'LC', 'DC', 'Cls', 'Loc', 'Bkg', 'Miss']
        self.type_color = {'Match': (10, 20, 190), 'LC': (100, 20, 190), 'DC': (30, 140, 140),
                           'Cls': (190, 20, 100), 'Loc': (190, 30, 30), 'Bkg': (50, 50, 50), 'Miss': (80, 20, 170)}
        self.cats = self.cocoGt.loadCats(self.cocoGt.getCatIds())

    def read_middle_file(self, middle_file: str) -> bool:
        """ Read a middle file and generate cocoGt and COCOdt

        Args:
            middle_file (str): Path of the middle file

        Returns:
            bool: True if cocoGt and cocoDT are generated.
        """
        if middle_file is not None:
            if os.path.isfile(middle_file):
                cocoGt = COCO(middle_file)
                _cocoDt = json.load(open(middle_file))['detections']
                cocoDt = cocoGt.loadRes(_cocoDt)
                self.is_evaluated = self.evaluation_check(cocoGt, cocoDt)
                if self.is_evaluated == False:
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

    def read_result_file(self, result_file: str) -> None:
        """Read a result file

        Args:
            result_file (str): Path of the result file
        """
        if os.path.isfile(result_file):
            self.results = json.load(open(result_file))['results']
            params_dict = json.load(open(result_file))['params']
            self.params = COCOParams(**params_dict)
            self.is_precision_calculated = True
            self.is_recall_calculated = True
            self.is_ap_calculated = True
            return True
        else:
            return False

    def visualize(self) -> None:
        """Viualize the results by drwawing bounding boxes, precision and recall curves, and APs.
        """
        if self.is_evaluated:
            self.draw_bounding_boxes()
        if self.is_precision_calculated:
            self.draw_precision_figs()
            self.pairplot(prec_or_recall='precision')
        if self.is_recall_calculated:
            self.draw_recall_figs()
            self.pairplot(prec_or_recall='recall')
        if self.is_ap_calculated:
            self.draw_ap_figs()

    def draw_precision_figs(self) -> None:
        """Draw precision figures.
        """

        precisions = self.results['precision']

        dir_fig_precision = os.path.join(
            self.result_dir, 'figures/precision')
        os.makedirs(dir_fig_precision, exist_ok=True)
        precision_lists = []
        for prec in precisions:

            # save each charts
            fig_title = prec['category'] + "_precision_ratio"

            p = draw_pi_chart(fig_title,
                              prec['ratio'].values(), prec['ratio'].keys())
            precision_lists.append(p)

        grid_precision = gridplot(
            [precision_lists])
        save(grid_precision, os.path.join(
            dir_fig_precision, 'precision_all.html'))

    def draw_recall_figs(self) -> None:
        """Draw recall figures
        """

        recalls = self.results['recall']

        dir_fig_recall = os.path.join(
            self.result_dir, 'figures', 'recall')
        os.makedirs(dir_fig_recall, exist_ok=True)

        recall_lists = []
        for rec in recalls:

            # save each charts
            fig_title = rec['category'] + "_precision_ratio"

            p = draw_pi_chart(fig_title,
                              rec['ratio'].values(), rec['ratio'].keys())
            recall_lists.append(p)

        grid_recall = gridplot(
            [recall_lists])
        save(grid_recall, os.path.join(
            dir_fig_recall, 'recall_all.html'))

    def draw_ap_figs(self) -> None:
        """Draw ap figures
        """

        dir_fig_ap = os.path.join(self.result_dir, 'figures', 'ap')
        os.makedirs(dir_fig_ap, exist_ok=True)

        aps = self.results['ap']
        pi_lists_all = []
        pr_curve_all = []
        pr_score_all = []

        for ap in aps:
            pi_fig_title = ap['category'] + '_' + ap['area'] + "_ap_ratio"
            pr_curve_fig_title = ap['category'] + \
                '_' + ap['area'] + "_pr_curve"
            pr_score_fig_title = ap['category'] + \
                '_' + ap['area'] + "_pr_score"

            # _ratio = {k: v for k, v in ap['ratio'].items() if v != 0}
            p = draw_pi_chart(pi_fig_title,
                              ap['ratio'].values(), ap['ratio'].keys())
            pi_lists_all.append(p)

            p = draw_pr_score(pr_score_fig_title,
                              ap['score'], ap['prec_raw'], ap['recall_raw'])
            pr_score_all.append(p)

            p = draw_pr_curve(pr_curve_fig_title,
                              ap['recall_raw'], ap['prec_raw'], self.params.recall_inter, ap['prec_inter'])
            pr_curve_all.append(p)

        grid_pi_all = gridplot(
            pi_lists_all, ncols=len(self.params.area_rng))
        save(grid_pi_all, os.path.join(dir_fig_ap, 'ap_ratio_all.html'))
        grid_pr_score_all = gridplot(
            pr_score_all, ncols=len(self.params.area_rng))
        save(grid_pr_score_all, os.path.join(dir_fig_ap, 'pr_score_all.html'))
        grid_pr_curve_all = gridplot(
            pr_curve_all, ncols=len(self.params.area_rng))
        save(grid_pr_curve_all, os.path.join(dir_fig_ap, 'pr_curve_all.html'))

    def draw_bounding_boxes(self) -> None:
        """Visualize all bounding boxes and types in images.
        """

        dir_TP = os.path.join(self.result_dir, 'draw_bbs', 'TP')
        dir_not_TP = os.path.join(self.result_dir, 'draw_bbs', 'not_TP')
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

    def pairplot(self, prec_or_recall):
        os.makedirs(os.path.join(self.result_dir, 'figures',
                                 prec_or_recall), exist_ok=True)
        if prec_or_recall == 'precision':
            obj_ids = self.cocoDt.getAnnIds()
            objs = self.cocoDt.loadAnns(obj_ids)
        elif prec_or_recall == 'recall':
            obj_ids = self.cocoGt.getAnnIds()
            objs = self.cocoGt.loadAnns(obj_ids)
        else:
            return False

        for obj in objs:
            del obj['attributes']
            del obj['iscrowd']
            del obj['segmentation']
            del obj['id']
            obj['count'] = obj['eval']['count']
            obj['type'] = obj['eval']['type']
            obj['iou'] = obj['eval']['iou']
            obj['bb_cx'] = obj['bbox'][0] + obj['bbox'][2]/2
            obj['bb_cy'] = obj['bbox'][1] + obj['bbox'][3]/2
            del obj['bbox']
            del obj['eval']

        df = pd.DataFrame(data=objs)
        pg = sns.pairplot(df, hue='type', hue_order=self.type)
        pg.savefig(os.path.join(self.result_dir, 'figures',
                                prec_or_recall, 'pairplot.png'))


if __name__ == '__main__':
    path_to_coco_dir = "example/data/"
    path_to_result_dir = "example/results/"
    path_to_middle_file = os.path.join(path_to_result_dir, 'middle_file.json')
    path_to_results_file = os.path.join(
        path_to_result_dir, 'final_results.json')
    path_to_image_dir = os.path.join(path_to_coco_dir, 'images')

    cocoVis = COCOVisualizer(path_to_middle_file, path_to_results_file,
                             path_to_result_dir, path_to_image_dir)
    cocoVis.visualize()
