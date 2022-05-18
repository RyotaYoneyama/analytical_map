
import os
from cocoEvaluator import COCOEvaluator
from cocoAnalizer import COCOAnalizer
import argparse
from params import cocoParams
import numpy as np


def get_arguments():
    parser = argparse.ArgumentParser(
        description='Analytical map')    # 2. パーサを作る

    parser.add_argument('gt', help='Ground truth coco path')
    parser.add_argument('dt', help='Detection coco path')
    parser.add_argument('result_dir', help='Result directory')
    parser.add_argument('image_dir', help='Image directory')
    args = parser.parse_args()

    return args


def eval_and_analyze(gt_path, dt_path, result_dir, image_dir, params: cocoParams):
    cocoEval = COCOEvaluator(gt_path, dt_path,
                             result_dir, image_dir, params)
    cocoEval.eval()
    cocoEval.visualize()
    middle_file_path = cocoEval.dump_middle_file_json()

    cocoAnal = COCOAnalizer(middle_file_path, result_dir, params)
    cocoAnal.precision_analyze()
    cocoAnal.recall_analyze()
    cocoAnal.ap_analyze()
    cocoAnal.dump_final_results_json()


def main():
    args = get_arguments()
    p = cocoParams(iou_thresh=0.5, iou_loc=0.2, recall_inter=np.arange(
        0, 1.01, 0.1), area_rng=np.array([[0, 1024], [1024, 9216], [9216, 10000000000.0]]))
    # p = cocoParams(iou_thresh=0.5, iou_loc=0.2, recall_inter=np.arange(0, 1.01, 0.1), area_rng=[])

    eval_and_analyze(args.gt, args.dt, args.result_dir, args.image_dir, p)


if __name__ == '__main__':
    main()
