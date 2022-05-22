import json
from pycocotools.coco import COCO
import os
import cv2


from analytical_map.params import cocoParams
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
        assert self.read_middle_file(middle_file)

        self.result_dir = result_dir
        self.image_dir = image_dir
        assert os.path.isdir(self.image_dir)

        assert self.read_result_file(results_file)

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

    def read_result_file(self, result_file: str) -> None:
        """Read a result file

        Args:
            result_file (str): Path of the result file
        """
        if os.path.isfile(result_file):
            self.results = json.load(open(result_file))['results']
            params_dict = json.load(open(result_file))['params']
            self.params = cocoParams(**params_dict)
            return True
        else:
            return False

    def draw_precision_figs(self) -> None:
        """Draw precision figures.
        """

        precisions = self.results['precision']

        dir_fig_precision = os.path.join(
            self.result_dir, 'figures', 'precision')
        img_lists = []
        for pres in precisions:

            # save each charts
            fig_title = pres['category'] + "_precision_ratio"

            _ratio = {k: v for k, v in pres['ratio'].items() if v != 0}

            draw_pi_chart(dir_fig_precision, fig_title,
                          _ratio.values(), _ratio.keys())
            img_lists.append(cv2.imread(os.path.join(
                dir_fig_precision, fig_title + '.png')))

        im_h = cv2.hconcat(img_lists)
        cv2.imwrite(os.path.join(dir_fig_precision, 'all_precision.png'), im_h)

    def draw_recall_figs(self) -> None:
        """Draw recall figures
        """

        recalls = self.results['recall']

        dir_fig_precision = os.path.join(
            self.result_dir, 'figures', 'recall')
        img_lists = []
        for rec in recalls:

            # save each charts
            fig_title = rec['category'] + "_precision_ratio"

            _ratio = {k: v for k, v in rec['ratio'].items() if v != 0}

            draw_pi_chart(dir_fig_precision, fig_title,
                          _ratio.values(), _ratio.keys())
            img_lists.append(cv2.imread(os.path.join(
                dir_fig_precision, fig_title + '.png')))

        im_h = cv2.hconcat(img_lists)
        cv2.imwrite(os.path.join(dir_fig_precision, 'all_recalls.png'), im_h)

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
            dir_fig_ap_each = os.path.join(dir_fig_ap, ap['category'])
            pi_fig_title = ap['category'] + '_' + ap['area'] + "_ap_ratio"
            pr_curve_fig_title = ap['category'] + \
                '_' + ap['area'] + "_pr_curve"
            pr_score_fig_title = ap['category'] + \
                '_' + ap['area'] + "_pr_score"

            _ratio = {k: v for k, v in ap['ratio'].items() if v != 0}
            draw_pi_chart(dir_fig_ap_each, pi_fig_title,
                          _ratio.values(), _ratio.keys())

            draw_pr_score(dir_fig_ap_each, pr_score_fig_title,
                          ap['score'], ap['prec_raw'], ap['recall_raw'])

            draw_pr_curve(dir_fig_ap_each, pr_curve_fig_title,
                          ap['recall_raw'], ap['prec_raw'], self.params.recall_inter, ap['prec_inter'])

            pi_lists_all.append(cv2.imread(os.path.join(
                dir_fig_ap_each, pi_fig_title + '.png')))

            pr_curve_all.append(cv2.imread(os.path.join(
                dir_fig_ap_each, pr_curve_fig_title + '.png')))

            pr_score_all.append(cv2.imread(os.path.join(
                dir_fig_ap_each, pr_score_fig_title + '.png')))

        pi_lists_all = convert_1d_to_2d(
            pi_lists_all, len(self.params.area_rng))
        pr_curve_all = convert_1d_to_2d(
            pr_curve_all, len(self.params.area_rng))
        pr_score_all = convert_1d_to_2d(
            pr_score_all, len(self.params.area_rng))

        pi_imgs = concat_tile(pi_lists_all)
        pr_curve_imgs = concat_tile(pr_curve_all)
        pr_score_imgs = concat_tile(pr_score_all)

        cv2.imwrite(os.path.join(dir_fig_ap,
                                 'all_ap_ratio.png'), pi_imgs)
        cv2.imwrite(os.path.join(dir_fig_ap,
                                 'all_pr_curve.png'), pr_curve_imgs)
        cv2.imwrite(os.path.join(dir_fig_ap,
                                 'all_pr_score.png'), pr_score_imgs)

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


if __name__ == '__main__':
    path_to_coco_dir = "example/data/"
    path_to_result_dir = "example/results/"
    path_to_middle_file = os.path.join(path_to_result_dir, 'middle_file.json')
    path_to_results_file = os.path.join(
        path_to_result_dir, 'final_results.json')
    path_to_image_dir = os.path.join(path_to_coco_dir, 'images')

    cocoVis = COCOVisualizer(path_to_middle_file, path_to_results_file,
                             path_to_result_dir, path_to_image_dir)
    cocoVis.draw_bounding_boxes()
    cocoVis.draw_precision_figs()
    cocoVis.draw_recall_figs()
    cocoVis.draw_ap_figs()