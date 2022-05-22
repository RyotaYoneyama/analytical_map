import json
from dataclasses import asdict
from typing import List
from pycocotools.coco import COCO
from analytical_map.params import cocoParams
import collections as cl
import os
import copy
from typing import List


def dump_middle_file_json(cocoGt: COCO, cocoDt: COCO, params: cocoParams, result_dir: str, middle_file: str = 'middle_file.json'):
    """Dump middle file

    Args:
        cocoGt (COCO): COCO ground truth
        cocoDt (COCO): COCO detections
        params (cocoParams): COCO params
        result_dir (str): Result directory path
        middle_file (str): Middle file name.  Defaults to 'middle_file.json'.

    Returns:
    """
    os.makedirs(result_dir, exist_ok=True)

    query_list = ["licenses", "info", "categories", "images",
                  "annotations", "detections", "params", "segment_info"]
    js = cl.OrderedDict()
    for i in range(len(query_list)):
        tmp = ""
        if query_list[i] == "categories":
            tmp = categories(cocoGt)
        if query_list[i] == "images":
            tmp = images(cocoGt)
        if query_list[i] == "annotations":
            tmp = annotations(cocoGt)
        if query_list[i] == "detections":
            tmp = detections(cocoDt)
        if query_list[i] == "params":
            tmp = param2dict(params)
        # save it
        js[query_list[i]] = tmp
    # write
    middle_file_path = os.path.join(result_dir, middle_file)
    fw = open(middle_file_path, 'w')
    json.dump(js, fw, indent=2)


def dump_final_results_json(cocoGt: COCO, params: cocoParams, results: list, result_dir: str, final_file: str = 'final_results.json'):
    """Dump final results

    Args:
        cocoGt (COCO): COCO ground truth
        params (cocoParams): COCO params
        results (list): result list
        result_dir (str): Result directory path
        final_file (str, optional): Final result file name. Defaults to 'final_results.json'.
    """

    os.makedirs(result_dir, exist_ok=True)

    query_list = ["licenses", "info", "categories", "params",
                  "results"]
    js = cl.OrderedDict()
    for i in range(len(query_list)):
        tmp = ""
        if query_list[i] == "categories":
            tmp = categories(cocoGt)
        if query_list[i] == "params":
            tmp = param2dict(params)
        if query_list[i] == "results":
            tmp = results
        js[query_list[i]] = tmp
    fw = open(os.path.join(result_dir, final_file), 'w')
    json.dump(js, fw, indent=2)


def images(cocoGt):
    img_ids = cocoGt.getImgIds()
    return cocoGt.loadImgs(ids=img_ids)


def categories(cocoGt):
    return cocoGt.loadCats(cocoGt.getCatIds())


def annotations(cocoGt):
    annIds = cocoGt.getAnnIds()
    return cocoGt.loadAnns(ids=annIds)


def detections(cocoDt):
    annIds = cocoDt.getAnnIds()
    return cocoDt.loadAnns(ids=annIds)


def param2dict(params):
    _params = copy.deepcopy(params)
    _params.recall_inter = _params.recall_inter.tolist()
    _params.area_rng = _params.area_rng.tolist()
    tmp = asdict(_params)
    return tmp
