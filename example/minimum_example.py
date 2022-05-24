
import argparse
import numpy as np
import os

from analytical_map import *


def get_arguments():
    parser = argparse.ArgumentParser(
        description='Analytical map')    # 2. パーサを作る

    parser.add_argument('gt', help='Ground truth coco path')
    parser.add_argument('dt', help='Detection coco path')
    parser.add_argument('result_dir', help='Result directory')
    parser.add_argument('image_dir', help='Image directory')
    args = parser.parse_args()

    return args


def main():
    args = get_arguments()
    p = cocoParams(iou_thresh=0.5, iou_loc=0.2, recall_inter=np.arange(
        0, 1.01, 0.1), area_rng=np.array([[0, 1024], [1024, 9216], [9216, 10000000000.0]]))
    # p = cocoParams(iou_thresh=0.5, iou_loc=0.2, recall_inter=np.arange(0, 1.01, 0.1), area_rng=[])
    cocoAnal = COCOAnalyzer(args.gt, args.dt,
                            args.result_dir, args.image_dir, p)
    cocoAnal.evaluate()
    cocoAnal.dump_middle_file_json('middle_file.json')
    cocoAnal.calculate()
    cocoAnal.dump_final_results_json('final_results.json')
    cocoAnal.visualize()


if __name__ == '__main__':
    main()
