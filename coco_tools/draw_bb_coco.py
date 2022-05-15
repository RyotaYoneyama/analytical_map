from pycocotools.coco import COCO
import cv2
import os
import argparse


def get_arguments():
    parser = argparse.ArgumentParser(
        description='Draw coco bounding boxes')    # 2. パーサを作る

    parser.add_argument('gt', help='ground truth')
    parser.add_argument('img_dir', help='img_dir')
    parser.add_argument('out_dir', help='out_dir')
    parser.add_argument('--inferences', '-i', help='inference inferences')
    args = parser.parse_args()

    return args


def draw_bb_coco(anno, img_dir, out_dir, inferences=None):
    cocoGt = COCO(anno)
    if inferences is not None:
        cocoInf = cocoGt.loadRes(inferences)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    img_ids = cocoGt.getImgIds()
    for img_id in img_ids:

        img = cocoGt.loadImgs(ids=img_id)[0]

        img_cv2 = cv2.imread(os.path.join(img_dir, img["file_name"]))
        annIds_gt = cocoGt.getAnnIds(imgIds=img['id'],  iscrowd=None)
        anns_gt = cocoGt.loadAnns(annIds_gt)
        for ann_gt in anns_gt:
            x_min = int(ann_gt["bbox"][0])
            y_min = int(ann_gt["bbox"][1])
            x_max = int(ann_gt["bbox"][0]) + int(ann_gt["bbox"][2])
            y_max = int(ann_gt["bbox"][1]) + int(ann_gt["bbox"][3])

            cv2.rectangle(img_cv2, (x_min, y_min),
                          (x_max, y_max), (0, 0, 255), thickness=2)

        if inferences is not None:
            annIds_inf = cocoInf.getAnnIds(imgIds=img['id'],  iscrowd=None)
            anns_inf = cocoInf.loadAnns(annIds_inf)

            for ann_inf in anns_inf:
                x_min = int(ann_inf["bbox"][0])
                y_min = int(ann_inf["bbox"][1])
                x_max = int(ann_inf["bbox"][0]) + int(ann_inf["bbox"][2])
                y_max = int(ann_inf["bbox"][1]) + int(ann_inf["bbox"][3])

                cv2.rectangle(img_cv2, (x_min, y_min),
                              (x_max, y_max), (255, 0, 0))

        cv2.imwrite(os.path.join(out_dir, img["file_name"]), img_cv2)


if __name__ == '__main__':
    args = get_arguments()
    draw_bb_coco(args.annotations, args.img_dir, args.out_dir, args.inferences)
