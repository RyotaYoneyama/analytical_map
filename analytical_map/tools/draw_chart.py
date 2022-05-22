import matplotlib.pyplot as plt
import os
import cv2
from nptyping import NDArray
import numpy as np


def draw_pi_chart(fig_dir: str, fig_title: str, values: list, labels: list) -> None:
    """ Draw a pi chart

    Args:
        fig_dir (str): Output directory path
        fig_title (str): Output figure title and image name
        values (list): Values
        labels (list): Labels
    """
    os.makedirs(fig_dir, exist_ok=True)
    fig_output_path = os.path.join(fig_dir, fig_title + '.png')
    fig_pie, ax_pie = plt.subplots(figsize=(4, 3))
    ax_pie.set_title(fig_title)
    ax_pie.pie(values, labels=labels,
               autopct="%1.1f %%")
    fig_pie.savefig(fig_output_path, pad_inches=0)


def draw_pr_score(fig_dir: str, fig_title: str, x_score: NDArray, y_pres: NDArray, y_recall: NDArray) -> None:
    """_summary_

    Args:
        fig_dir (str): Output directory path
        fig_title (str): Output figure title and image name
        x_score (NDArray): Scores 
        y_pres (NDArray): Precisions sorted by score
        y_recall (NDArray): Recalls sorted by score
    """
    os.makedirs(fig_dir, exist_ok=True)
    fig_output_path = os.path.join(fig_dir, fig_title + '.png')

    fig_score, ax_score = plt.subplots()
    ax_score.set_title(fig_title)
    ax_score.plot(x_score, y_pres, color='red')
    ax_score.plot(x_score, y_recall, color='green')
    ax_score.set_xlim(-0.05, 1.05)
    ax_score.set_ylim(-0.05, 1.05)
    ax_score.legend(["Precision", "Recall"])
    ax_score.set_xlabel('Score')
    fig_score.savefig(fig_output_path, figsize=(4, 3))


def draw_pr_curve(fig_dir: str, fig_title: str, recall: NDArray, precision: NDArray, recall_inter: NDArray, precision_inter: NDArray) -> None:
    """_summary_

    Args:
        fig_dir (str): Output directory path
        fig_title (str): Output figure title and image name
        recall (NDArray): Recall sorted by score
        precision (NDArray): Precision sorted by score
        recall_inter (NDArray): Recall for integral  
        precision_inter (NDArray): Precision for integral 
    """

    os.makedirs(fig_dir, exist_ok=True)
    fig_output_path = os.path.join(fig_dir, fig_title + '.png')

    fig_pr, ax_pr = plt.subplots()
    ax_pr.set_title(fig_title)
    ax_pr.plot(recall, precision, marker='.', color='royalblue')
    ax_pr.plot(recall_inter, precision_inter,
               marker='x', linewidth=0, color='orange')
    ax_pr.set_xlabel('Precision')
    ax_pr.set_xlabel('Recall')
    ax_pr.legend(["PR_raw", "PR_inter"])
    ax_pr.set_xlim(-0.05, 1.05)
    ax_pr.set_ylim(-0.05, 1.05)
    fig_pr.savefig(fig_output_path, figsize=(4, 3))


def concat_tile(im_list_2d: list) -> NDArray:
    """_summary_

    Args:
        im_list_2d (list): 2D list of imgs: i.e.,[[img1, img2,], [img3, img4]].

    Returns:
        NDArray: image
    """
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])


def convert_1d_to_2d(l: list, cols: int) -> list:
    """Convert 1D list to 2D

    Args:
        l (list): 1D list
        cols (int): The number of cols

    Returns:
        list: 2D list
    """

    return [l[i:i + cols] for i in range(0, len(l), cols)]
