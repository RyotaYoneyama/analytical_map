
import os
from cocoEvaluator import COCOEvaluator
from cocoAnalizer import COCOAnalizer
import argparse


def get_arguments():
    parser = argparse.ArgumentParser(
        description='Analytical map')    # 2. パーサを作る

    parser.add_argument('gt', help='Ground truth coco path')
    parser.add_argument('dt', help='Detection coco path')
    parser.add_argument('result_dir', help='Result directory')
    parser.add_argument('image_dir', help='Image directory')
    args = parser.parse_args()

    return args


def eval_and_analyze(gt_path, dt_path, result_dir, image_dir):
    cocoEval = COCOEvaluator(gt_path, dt_path,
                             result_dir, image_dir)
    cocoEval.eval()
    cocoEval.visualize()
    middle_file_path = cocoEval.dump_middle_file_json()

    cocoAnal = COCOAnalizer(middle_file_path, result_dir)
    cocoAnal.precision_analyze()
    cocoAnal.recall_analyze()
    cocoAnal.ap_analyze()
    cocoAnal.dump_final_results_json()


def main():
    args = get_arguments()
    eval_and_analyze(args.gt, args.dt, args.result_dir, args.image_dir)


if __name__ == '__main__':
    main()
