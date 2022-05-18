import cv2
from pycocotools.coco import COCO
from pycocotools.coco import COCO
import cv2
import os
import argparse


def get_arguments():
    parser = argparse.ArgumentParser(
        description='Draw coco bounding boxes')    # 2. パーサを作る

    parser.add_argument('gt', help='ground truth')
    parser.add_argument('dt', help='dt')
    parser.add_argument('img_dir', help='img_dir')
    parser.add_argument('out_dir', help='out_dir')
    args = parser.parse_args()

    return args


def draw_bb_coco(gt_path, dt_path, img_dir, out_dir):
    cocoGt = COCO(gt_path)
    cocoDt = cocoGt.loadRes(dt_path)
    categories = cocoGt.loadCats(cocoGt.getCatIds())

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    img_ids = cocoGt.getImgIds()
    for img_id in img_ids:

        img = cocoGt.loadImgs(ids=img_id)[0]

        img_cv2 = cv2.imread(os.path.join(img_dir, img["file_name"]))
        img_cv2_gt = img_cv2.copy()
        img_cv2_dt = img_cv2.copy()

        Ids_gts = cocoGt.getAnnIds(imgIds=img['id'],  iscrowd=None)
        gts = cocoGt.loadAnns(Ids_gts)
        for gt in gts:
            x_min = int(gt["bbox"][0])
            y_min = int(gt["bbox"][1])
            x_max = int(gt["bbox"][0]) + int(gt["bbox"][2])
            y_max = int(gt["bbox"][1]) + int(gt["bbox"][3])

            cv2.rectangle(img_cv2_gt, (x_min, y_min),
                          (x_max, y_max), (0, 0, 255), thickness=2)
            category_id = gt['category_id']
            for cat in categories:
                if cat['id'] == category_id:
                    name = cat['name']
                    break
            cv2.putText(img_cv2_gt,
                        text=name+'_id_' + str(gt['id']),
                        org=(x_min, y_min-5),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(0, 0, 0),
                        thickness=1,
                        lineType=cv2.LINE_4)

        Ids_dt = cocoDt.getAnnIds(imgIds=img['id'],  iscrowd=None)
        dts = cocoDt.loadAnns(Ids_dt)
        for dt in dts:
            x_min = int(dt["bbox"][0])
            y_min = int(dt["bbox"][1])
            x_max = int(dt["bbox"][0]) + int(dt["bbox"][2])
            y_max = int(dt["bbox"][1]) + int(dt["bbox"][3])
            cv2.rectangle(img_cv2_dt, (x_min, y_min),
                          (x_max, y_max), (255, 0, 0))
            category_id = dt['category_id']
            for cat in categories:
                if cat['id'] == category_id:
                    name = cat['name']
                    break
            cv2.putText(img_cv2_dt,
                        text=name+'_id_' + str(dt['id']),
                        org=(x_min, y_min-5),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(0, 0, 0),
                        thickness=1,
                        lineType=cv2.LINE_4)
            cv2.putText(img_cv2_dt,
                        text=str(round(dt['score'], 2)),
                        org=(x_min, y_min+15),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(0, 0, 0),
                        thickness=1,
                        lineType=cv2.LINE_4)

        img_cv2_gt_dt = cv2.hconcat([img_cv2_gt, img_cv2_dt])
        cv2.imwrite(os.path.join(out_dir, img["file_name"]), img_cv2_gt_dt)


if __name__ == '__main__':
    args = get_arguments()
    draw_bb_coco(args.gt, args.dt,  args.img_dir, args.out_dir)
