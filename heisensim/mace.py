from scipy.spatial.ckdtree import cKDTree
from heisensim.spin_model import SpinModelSym, XX, DipoleCoupling
from heisensim.spin_half import expect, sx
import numpy as np
import pandas as pd


def pos_for_mace(positions, spin_number, cluster_size):
    tree = cKDTree(data=positions)
    pos_i = positions[spin_number]
    _, pos_indices = tree.query(pos_i, cluster_size)
    pos = positions[pos_indices]
    return pos
