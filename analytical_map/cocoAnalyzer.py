import numpy as np
import os
from nptyping import NDArray


from analytical_map.params import COCOParams
from analytical_map.tools.dump_json import dump_middle_file_json as _dump_middle_file_json
from analytical_map.cocoEvaluator import COCOEvaluator
from analytical_map.cocoCalculator import COCOCalculator
from analytical_map.cocoVisualizer import COCOVisualizer


class COCOAnalyzer(COCOEvaluator, COCOCalculator, COCOVisualizer):
    def __init__(self, cocoGt_file: str, cocoDt_file: str, result_dir: str, image_dir: str, params: COCOParams) -> None:
        """Init

        Args:
            cocoGt_file (str): COCO ground truth path
            cocoDt_file (str): COCO detection file path
            result_dir (str): Output path
            image_dir (str): Image directory path
            params (COCOParams): Parameters for evaluation
        """
        # Input
        self.cocoGt = None
        self.cocoDt = None
        self.cats = None

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
