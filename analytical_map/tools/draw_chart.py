from nptyping import NDArray
from bokeh.plotting import figure
from bokeh.palettes import Category20c
import pandas as pd
from bokeh.transform import cumsum
from math import pi

TOOLS = "pan,wheel_zoom,box_zoom,box_select,crosshair,reset,save"


def draw_pi_chart(fig_title: str, values: list, labels: list) -> None:
    """ Draw a pi chart

    Args:
        fig_title (str): Output figure title and image name
        values (list): Values
        labels (list): Labels
    Returns:
        p (Figure): bokeh figure
    """
    p = figure(height=350, title=fig_title, toolbar_location=None,
               tools=TOOLS, tooltips="@type: @value", x_range=(-0.5, 1.0))

    data = pd.Series(values, labels).reset_index(
        name='value').rename(columns={'index': 'type'})
    data['angle'] = data['value']/data['value'].sum() * 2*pi
    data['color'] = Category20c[len(values)]
    p.wedge(x=0, y=1, radius=0.4,
            start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
            line_color="white", fill_color='color', legend_field='type', source=data)

    p.axis.axis_label = None
    p.axis.visible = False
    p.grid.grid_line_color = None
    return p


def draw_pr_score(fig_title: str, x_score: NDArray, y_prec: NDArray, y_recall: NDArray) -> None:
    """_summary_

    Args:
        fig_title (str): Output figure title and image name
        x_score (NDArray): Scores
        y_prec (NDArray): Precisions sorted by score
        y_recall (NDArray): Recalls sorted by score
    Returns:
        p (Figure): bokeh figure
    """
    p = figure(title=fig_title,
               width=450,
               toolbar_location="right",
               tools=TOOLS,
               tooltips="Data point @x has the value @y",
               x_axis_label="Score",
               y_axis_label="Precisio or Recall")

    p.line(x_score, y_prec, legend_label="Precision",
           line_color="red",  line_dash='dashed')
    p.circle(x_score, y_prec, color='red', line_width=5)
    p.line(x_score, y_recall, legend_label="Recall",
           line_color="green",   line_dash='dashed')
    p.circle(x_score, y_recall, color='green', line_width=5)

    p.legend.location = "top_left"

    return p


def draw_pr_curve(fig_title: str, recall: NDArray, precision: NDArray, recall_inter: NDArray, precision_inter: NDArray) -> None:
    """_summary_

    Args:
        fig_title (str): Output figure title and image name
        recall (NDArray): Recall sorted by score
        precision (NDArray): Precision sorted by score
        recall_inter (NDArray): Recall for integral
        precision_inter (NDArray): Precision for integral
    Returns:
        p (Figure): bokeh figure
    """

    TOOLS = "pan,wheel_zoom,box_zoom,box_select,crosshair,reset,save"
    p = figure(title=fig_title,
               toolbar_location="right",
               tools=TOOLS,
               tooltips="Data point @x has the value @y",
               x_axis_label="Score",
               y_axis_label="Precisio or Recall")

    p.line(recall, precision, legend_label="Raw",
           line_color="red",  line_dash='dashed')
    p.circle(recall, precision, color='red', line_width=5)
    p.line(recall_inter, precision_inter, legend_label="Recall",
           line_color="green",   line_dash='dashed')
    p.circle(recall_inter, precision_inter, color='green', line_width=5)

    p.legend.location = "top_left"
    return p
