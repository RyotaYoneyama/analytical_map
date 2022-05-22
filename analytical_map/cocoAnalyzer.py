import numpy as np
import os
from nptyping import NDArray


from analytical_map.params import cocoParams
from analytical_map.tools.dump_json import dump_middle_file_json as _dump_middle_file_json
from analytical_map.cocoEvaluator import COCOEvaluator
from analytical_map.cocoCalculator import COCOCalculator
from analytical_map.cocoVisualizer import COCOVisualizer


class COCOAnalyzer(COCOEvaluator, COCOCalculator, COCOVisualizer):
    def __init__(self, cocoGt_file: str, cocoDt_file: str, result_dir: str, image_dir: str, params: cocoParams) -> None:
        """Init

        Args:
            cocoGt_file (str): COCO ground truth path
            cocoDt_file (str): COCO detection file path
            result_dir (str): Output path
            image_dir (str): Image directory path
            params (cocoParams): Parameters for evaluation
        """
        # Input
        self.cocoGt = None
        self.cocoDt = None
        assert self.init_coco(cocoGt_file, cocoDt_file)

        self.result_dir = result_dir
        self.image_dir = image_dir
        assert os.path.isdir(self.image_dir)

        # Fixed variables
        self.type = ['Match', 'LC', 'DC', 'Cls', 'Loc', 'Bkg', 'Miss']
        self.type_order = {'Match': 0, 'LC': 1, 'DC': 1,
                           'Cls': 2, 'Loc': 3, 'Bkg': 4, 'Miss': 4, None: 5}
        self.type_color = {'Match': (10, 20, 190), 'LC': (100, 20, 190), 'DC': (30, 140, 140),
                           'Cls': (190, 20, 100), 'Loc': (190, 30, 30), 'Bkg': (50, 50, 50), 'Miss': (80, 20, 170)}
        self.cats = self.cocoGt.loadCats(self.cocoGt.getCatIds())
        self.results = {'precision': [], 'recall': [], 'ap': []}
        self.area_all = [0, 10000000000.0]

        self.is_evaluated = False
        self.is_ap_calculated = False
        self.is_precision_calculated = False
        self.is_recall_calculated = False

        # User variables
        self.params = params
        self.params.area_rng = np.insert(
            params.area_rng, 0, self.area_all, axis=0)

    def evaluate(self, middle_file: str = 'middle_file.json') -> None:
        """Evaluate all images and generate a middle file.

        Args:
            middle_file (str, optional): A name of the middle file. Defaults to 'middle_file.json'.
        """
        self.eval()
        self.dump_middle_file_json(middle_file=middle_file)
        self.is_evaluated = True

    def calculate(self, final_file: str = 'final_results.json') -> None:
        """Calculate precisions, recalls, and APs, and dump them as final results.

        Args:
            final_file (str, optional): A name of the final results. Defaults to 'final_results.json'.
        """
        if self.evaluation_check(self.cocoGt, self.cocoDt) == False:
            return False
        self.precision_calculate()
        self.recall_calculate()
        self.ap_calculate()
        self.dump_final_results_json(final_file='final_results.json')

        return True

    def visualize(self) -> None:
        """Viualize the results by drwawing bounding boxes, precision and recall curves, and APs.
        """
        if self.is_evaluated:
            self.draw_bounding_boxes()
        if self.is_precision_calculated:
            self.draw_precision_figs()
        if self.is_recall_calculated:
            self.draw_recall_figs()
        if self.is_ap_calculated:
            self.draw_ap_figs()


if __name__ == '__main__':
    path_to_coco_dir = "example/data/"
    path_to_result_dir = "example/results/"
    path_to_image_dir = os.path.join(path_to_coco_dir, 'images')
    path_to_gt = os.path.join(path_to_coco_dir, 'coco', 'gt.json')
    path_to_dt = os.path.join(path_to_coco_dir, 'coco', 'dt.json')

    p = cocoParams(iou_thresh=0.5, iou_loc=0.2,
                   area_rng=np.array([[0, 0], [10, 10000000000]]))
    cocoAnal = COCOAnalyzer(path_to_gt, path_to_dt,
                            path_to_result_dir, path_to_image_dir, p)
    cocoAnal.evaluate(middle_file='middle_file.json')
    cocoAnal.calculate(final_file='final_results.json')
    cocoAnal.visualize()
