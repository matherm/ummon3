# -*- coding: utf-8 -*-
# @Author: Daniel Dold, Markus Käppeler
# @Date:   2019-11-20 10:08:49
# @Last Modified by:   Markus Käppeler
# @Last Modified time: 2019-12-06 14:12:55
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.sparse import csr_matrix
from scipy.spatial import HalfspaceIntersection, ConvexHull
from scipy.spatial.qhull import QhullError
import logging
import os
import sklearn.metrics
from .base import *

__all__ = ['IoU', 'MeanDistanceError']

__log = logging.getLogger('geometric_metrics_log')
os.makedirs("./logfile/", exist_ok=True)
# create file handler which logs even debug messages
fh = logging.FileHandler("./logfile/" + 'geometric_metrics.log')
fh.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
# add the handlers to the logger
__log.addHandler(fh)
__log.propagate = True
#######
# activate logging with "logging.getLogger().setLevel(0)"
#######


def halfspace_representation(cuboid: dict) -> np.array:
    """compute half space representation from a cuboid (bounding cuboid)

    Args:
        cuboid (dict): contains the cuboid parameters.
                    Format: 'c' -> center of cuboid array[x,y, ... n]
                            'd' -> dimension of cuboid array[length,width, ... n]
                            'r' -> rotation of cuboid as 3x3 rot matrix

    Returns:
        np.array: cuboid in half space representation (shape [4xn+1])
    """
    # create normal vectors from a sparse matrix
    p_dim = cuboid['d'].shape[0]
    row = np.arange(2 * p_dim)
    col = np.array(list(np.arange(p_dim)) * 2)
    data = np.array([-1, 1]).repeat(p_dim)
    norm_vec = csr_matrix((data, (row, col)), shape=(p_dim * 2, p_dim)).toarray()  # 4x2 or 6x3
    # Rotation of axis aligned normal vectors
    norm_vec_rot = np.matmul(norm_vec, cuboid['r'].T)  # 4x2 or 6x3

    # compute d parameter of plane representation
    # p1 and p2 span the bounding cuboid volume (diagonal to each other)
    p1 = cuboid['c'] + np.matmul(-(cuboid['d'] / 2), cuboid['r'].T)
    p2 = cuboid['c'] + np.matmul((cuboid['d'] / 2), cuboid['r'].T)
    # following possible because of special normal matrix order
    d1 = -np.matmul(norm_vec_rot[0:p_dim, :], p1.reshape(-1, 1))
    d2 = -np.matmul(norm_vec_rot[p_dim:p_dim * 2, :], p2.reshape(-1, 1))
    d = np.concatenate((d1, d2), axis=0)
    halfspaces = np.concatenate((norm_vec_rot, d), axis=1)
    return halfspaces


def intersection(cuboid1: dict, cuboid2: dict) -> float:
    """compute the intersection of two arbitrary cuboids

    Args:
        cuboid1 (dict): cuboid 1 parameters.
                    Format: 'c' -> center of cuboid array[x,y, ... n]
                            'd' -> dimension of cuboid array[length,width, ... n]
                            'r' -> rotation of cuboid as 3x3 rot matrix
        cuboid2 (dict): cuboid 2 parameters. Same data structure

    Returns:
        float: the volume of the intersection
    """
    halfspaces1 = halfspace_representation(cuboid1)
    halfspaces2 = halfspace_representation(cuboid2)
    halfspaces = np.concatenate((halfspaces1, halfspaces2), axis=0)
    # compute most likely point which is in the intersection area
    fp_vec = cuboid2['c'] - cuboid1['c']
    scale = cuboid1['d'] / (cuboid1['d'] + cuboid2['d'])
    feasible_point = fp_vec * scale + cuboid1['c']
    # run computation
    try:
        hs = HalfspaceIntersection(halfspaces, feasible_point)
        hull = ConvexHull(hs.intersections)
    except QhullError as e:
        __log.debug("no intersection found. ERROR msg: {}".format(e))
        return 0
    return hull.volume


def iou(cuboid1: dict, cuboid2: dict) -> float:
    intersect = intersection(cuboid1, cuboid2)
    union = np.prod(cuboid1['d']) + np.prod(cuboid2['d']) - intersect
    return intersect / union


def find_correspondences(outputs: list, targets: list, threshold: float) -> np.ndarray:
    """
    Computes the y_pred/y_true matrix by finding the correct target for each output. Each output counts to it's
    largest overlapping target. Multiple detections of the same object are considered as a false detection.
    To count as a correct prediction (True, True) the iou between an output and a target has to exceed
    the given threshold.
    If a target has not matching output, it count as a FN (False, True).
    If a output has not matching target, it counts as an FP (True, False).

    Args:
        outputs (list): list of predicted cuboids
        targets (list): list of target cuboids
    Returns:
        np.ndarray (N, 2): array with y_pred(N, 0) and y_true(N, 1), either True or False.
    """
    y_pred_true = []

    num_target = len(targets)
    num_output = len(outputs)

    # calculate all ious
    ious = np.zeros((num_target, num_output))
    for i_target, cuboid_target in enumerate(targets):
        for i_output, cuboid_output in enumerate(outputs):
            ious[i_target, i_output] = iou(cuboid_target, cuboid_output)

    targets_found = set()  # indices of found targets
    output_found = set()  # indices of found predictions
    iou_argsort_target = np.argsort(ious, axis=0)  # sort that each output counts to it's largest overlap target
    for target_i in reversed(range(num_output)):  # iterate reverse to start with largest iou (from argsort)
        for pred_i in range(num_target):
            target_max_i = iou_argsort_target[target_i, pred_i]

            # to count as correct detection area of overlap must exceed the threshold
            if ious[target_max_i, pred_i] > threshold and pred_i not in output_found:
                targets_found.add(target_max_i)
                output_found.add(i_output)
                y_pred_true += [[1, 1]]  # tp
                break

    targets_i = set(range(num_output))
    output_i = set(range(num_target))
    targets_not_matching = targets_i - targets_found  # targets with no matching detection
    y_pred_true += [[0, 1] for _ in range(len(targets_not_matching))]  # fn
    output_not_matching = output_i - output_found  # predictions with not matching targets
    y_pred_true += [[1, 0] for _ in range(len(output_not_matching))]  # fp

    return np.array(y_pred_true, dtype=bool)


class GeometricMetrics(OnlineMetric):
    def __call__(self, output: list, target: list):
        results = [self.func(c1, c2) for c1, c2 in zip(output, target)]
        return results

    @classmethod
    def __repr__(cls):
        return cls.__name__


class BinaryAPSklearn(GeometricMetrics):
    """
    Calculates the AP score with the function from Sklearn.
    At the moment it works only for binary detection without confidence scores.
    Usage:  ap = BinaryAPSklearn()
            ap(output_batch: list, target_batch: list) # list of list with cuboids
    Cuboid parameters (dict):Format:'c' -> center of cuboid array[x,y, ... n]
                                    'd' -> dimension of cuboid array[length,width, ... n]
                                    'r' -> rotation of cuboid as 3x3 rot matrix
    Attributes:
        func (TYPE): function used in parent class
    """

    def __init__(self):
        self.func = self.average_precision_sklearn

    @staticmethod
    def average_precision_sklearn(output: list, target: list):
        y_pred_true = find_correspondences(output, target, 0.5)
        y_pred, y_true = y_pred_true[:, 0], y_pred_true[:, 1]
        return sklearn.metrics.average_precision_score(y_true, y_pred)


class IoU(GeometricMetrics):
    """Compute the Intersection over Union (IoU) of two arbitrary 2d-/ 3d cuboids
    Usage:  iou = IoU()
            iou(cuboids_1: list, cuboids_2: list)  # cuboids wrapped in list
    Cuboid parameters (dict):Format:'c' -> center of cuboid array[x,y, ... n]
                                    'd' -> dimension of cuboid array[length,width, ... n]
                                    'r' -> rotation of cuboid as 3x3 rot matrix
    Attributes:
        func (TYPE): function used in parent class
    """

    def __init__(self):
        self.func = self.intersection_over_union


    @staticmethod
    def intersection_over_union(output: dict, target: dict):
        return iou(output, target)


class MeanDistanceError(GeometricMetrics):
    """Compute the geometric distance of two cuboids
    Usage:  dist = MeanDistanceError()
            dist(cuboids_1: list, cuboids_2: list) # cuboids wrapped in list
    Cuboid parameters (dict):Format:'c' -> center of cuboid array[x,y, ... n]
                                    'd' -> dimension of cuboid array[length,width, ... n]
                                    'r' -> rotation of cuboid as 3x3 rot matrix
    Attributes:
        func (TYPE): function used in parent class
    """

    def __init__(self):
        self.func = self.mean_distance_error

    @staticmethod
    def mean_distance_error(output: dict, target: dict):
        error_vec = output['c'] - target['c']
        error = np.sqrt(np.dot(error_vec, error_vec))
        return error


if __name__ == '__main__':
    # usage example
    # prepare demo data
    out = [dict(c=np.array([0., 1., 2.]),
                d=np.array([4., 8., 10.]),
                r=Rotation.from_euler('xyz', [45, 10, 30], degrees=True).as_dcm()),
           dict(c=np.array([0., 1., 2.]),
                d=np.array([4., 8., 10.]),
                r=Rotation.from_euler('xyz', [45, 10, 30], degrees=True).as_dcm())
           ]
    target = [dict(c=np.array([0., 1., 2.]),
                   d=np.array([4., 8., 10.]),
                   r=Rotation.from_euler('xyz', [45, 10, 30], degrees=True).as_dcm()),
              dict(c=np.array([0., 1.5, 2.]),
                   d=np.array([8., 8., 10.]),
                   r=Rotation.from_euler('xyz', [45, 20, 30], degrees=True).as_dcm())
              ]
    # usage example
    m = MeanDistanceError()
    m.test = "nachtr"
    metrics = [m, MeanDistanceError(), IoU()]
    result = {repr(m): m(out, target) for m in metrics}
    print(result)

    metrics = [BinaryAPSklearn()]
    result = {repr(m): m([[out[0]]], [[target[0]]]) for m in metrics}
    print(result)

    metrics = [BinaryAPSklearn()]
    result = {repr(m): m([out], [target]) for m in metrics}
    print(result)
