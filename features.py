# Stone colour            3 Player stones; oppo. stones; empty
# Ones                    1 Constant plane
# Turns since last move   8 How many turns since a move played
# Liberties               8 Number of liberties
# Zeros                   1 Constant plane

from pyexpat import features
import numpy as np
from go import Go


def color_stone_features(board, currentColor):
    # blank, this color, opponent color
    if currentColor == 1:
        features = [board == 0, board == 1, board == -1]
    else:
        features = [board == 0, board == -1, board == 1]
    return features


def ones_features():
    features = [np.ones((19, 19))]
    return features


# def turns_since_last_move_features(board):
#     features = [np.zeros(19, 19, dtype=np.int8)]
#     return features


def liberties_features(liberty, length=8):
    features = []
    for i in range(1, length + 1):
        features.append(liberty == i)
    return features


def zeros_features():
    features = [np.zeros(19, 19, dtype=np.int8)]
    return features


def getAllFeatures(go, currentColor):
    board = go.board
    liberty = go.liberty
    features = color_stone_features(board, currentColor) + ones_features() + liberties_features(liberty)
    return np.array(features)
