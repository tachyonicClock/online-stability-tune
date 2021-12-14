import re
import pandas as pd
import gzip
import json
import natsort
from typing import Any, List, Tuple


def random_series(classes_per_experiences, experiences):
    total_classes = classes_per_experiences * experiences
    prob = [1/x for x in range(classes_per_experiences, total_classes+1, classes_per_experiences)]
    indices = range(1, experiences+1)
    return pd.Series(prob, index=indices, name="Random")


def style_builder(*line_styles: Tuple[Any, int]):
    styles = []
    for (line_style, number) in list(line_styles):
        styles += [line_style] * number
    return styles


def remove_xaxis_ticks(ax):
    for tick in ax.xaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)

# Multiply a tuple ratio by a multiplier mult
def xy_mult(ratio: Tuple[float, float], mult: float):
    return (ratio[0] * mult, ratio[1] * mult)

