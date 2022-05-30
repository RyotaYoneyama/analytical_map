from dataclasses import dataclass
import numpy as np


@dataclass
class COCOParams:

    iou_thresh: float = 0.5
    score_thresh: float = 0.0001
    iou_loc: float = 0.2
    recall_inter: np.arange = np.arange(0, 1.01, 0.1)
    area_rng: np.array = np.array(
        [[0, 1024], [1024, 9216], [9216, 10000000000.0]])
