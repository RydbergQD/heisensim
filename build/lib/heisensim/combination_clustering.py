from __future__ import annotations

import math
from abc import ABC
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist


def nan_distance_matrix(positions):
    dist = distance_matrix(positions, positions)
    np.fill_diagonal(dist, np.nan)
    return dist


class PositionChooser(ABC):
    def choose_position(self, positions: list | np.ndarray) -> int:
        """Choose a position"""
        pass


class TrivialPositionChooser(PositionChooser):
    def choose_position(self, positions):
        return 0


class ClosestPairChooser(PositionChooser):
    def choose_position(self, positions):
        dist = nan_distance_matrix(positions)
        idx = np.unravel_index(np.nanargmin(dist), dist.shape)
        return idx[0]  # choose only one particle of the pair


@dataclass
class ExtremalPositionChooser(PositionChooser):
    origin: Optional[np.ndarray] = None

    def choose_position(self, positions):
        if self.origin is not None:
            positions = positions - self.origin
        dist_to_origin = np.linalg.norm(positions, axis=1)
        return np.argmin(dist_to_origin)


class OrderDeterminator(ABC):
    def determine_order(self, positions, origin):
        pass


class TrivialOrderer(OrderDeterminator):
    def determine_order(self, positions, origin: int = 0):
        return np.arange(len(list(positions)))


@dataclass
class ClosestOrderer(OrderDeterminator):
    def determine_order(self, positions, origin: int = 1):
        dist_to_origin = cdist(positions, [positions[origin, :]])[:, 0]
        return np.argsort(dist_to_origin)


@dataclass
class PairCombinator(ABC):
    position_chooser: PositionChooser = TrivialPositionChooser()
    order_determinator: OrderDeterminator = TrivialOrderer()
    cut: Optional[int] = None

    def _cut(self, positions):
        if self.cut is None:
            return len(list(positions))
        else:
            return self.cut

    def all_pairs(self, positions: np.ndarray):
        if len(positions) < 2:
            yield []
            return
        else:
            origin = self.position_chooser.choose_position(positions)
            order = self.order_determinator.determine_order(positions, origin)[1 : self._cut(positions)]
            for other in order:
                pair = positions[[origin, other]]
                new_positions = np.delete(positions, [origin, other], axis=0)
                for rest in self.all_pairs(new_positions):
                    yield [pair] + rest

    def all_pairs_variable_cut(self, positions: np.ndarray, cut=10):
        if len(positions) < 2:
            yield []
            return
        else:
            origin = self.position_chooser.choose_position(positions)
            order = self.order_determinator.determine_order(positions, origin)[1 : max(2, cut)]
            for other in order:
                pair = positions[[origin, other]]
                new_positions = np.delete(positions, [origin, other], axis=0)
                for rest in self.all_pairs_variable_cut(new_positions, cut=cut-1):
                    yield [pair] + rest


def closest_pair(pair_combos):
    return min(pair_combos, key=get_total_length)


def get_total_length(pairs: list):
    """Calculate the sum of the distances of each pair."""
    return sum(map(lambda pair: math.dist(*pair), pairs))
