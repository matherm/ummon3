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
from enum import Enum
from sklearn.preprocessing import label_binarize

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


class Order(Enum):
    OVERLAP_OUTPUT_TO_TARGET = 1  # each output matches with it's largest overlapping target, variant to input order
    OVERLAP_TARGET_TO_OUTPUT = 2  # each target matches with it's largest overlapping output, variant to input order
    CONFIDENCE_SCORE = 3  # matching with descending confidence, outputs with high confidence score are matched first,
    # invariant to input order
    OVERLAP = 4  # matching with descending overlap => matches with large overlap are matched first,
    # invariant to input order


def to_one_vs_all(classes: list, class_id: int, confidence_score=1.0) -> list:
    """
    Calculates one vs all vector (similar to sklearn.preprocessing.label_binarize, but only for one value, not array).

    Args:
        classes (list): list of class ids
        class_id (int): class id to encode
        confidence_score (float): confidence score to encode class id with
    Returns:
        list (num_classes,): one vs all list
    """
    l = [0.0 for _ in range(len(classes))]
    l[classes.index(class_id)] = confidence_score
    return l


def find_correspondences(outputs: list, targets: list, classes: list, threshold: float, order: Order) -> list:
    """
    Computes the y_pred and y_score matrix by matching target and output with a given order.
    Multiple detections of the same object are considered as a false detection.
    To count as a correct prediction (True, True) the iou between an output and a target has to exceed
    the given threshold.
    If a target has not matching output, it count as a FN (False, True).
    If a output has not matching target, it counts as an FP (True, False).

    Args:
        outputs (list): list of predicted cuboids (dict)
        targets (list): list of target cuboids (dict)
        classes (list): list of classes (int)
        threshold (float): threshold for iou to count as correct detection
        order (Order): order to match output and targets, if set to Order.CONFIDENCE_SCORE, output cuboids must
            contain confidence_score.
    Returns:
        tuple (2,): tuple of y_score and y_true, each with shape (N, num_classes) or of shape (N,) if binary (one label)
            , True binary labels or binary label indicators. Same format as params for sklearn average_precision_score.
    """
    y_true = []
    y_score = []

    num_classes = len(classes)

    for class_id in classes:
        # match outputs and targets with same class id
        outputs_class = [c for c in outputs if c['class_id'] == class_id]
        targets_class = [c for c in targets if c['class_id'] == class_id]

        num_output = len(outputs_class)
        num_target = len(targets_class)

        # calculate all ious
        ious = np.zeros((num_target, num_output))
        for i_target, cuboid_target in enumerate(targets_class):
            for i_output, cuboid_output in enumerate(outputs_class):
                ious[i_target, i_output] = iou(cuboid_target, cuboid_output)

        # Assignment of output - target to given order
        # each argsort sorts negative values to start with largest value
        if order == Order.CONFIDENCE_SCORE:
            confidence_scores = np.array([cuboid['confidence_score'] for cuboid in outputs_class])
            confidence_scores_argsort = np.argsort(-confidence_scores)
        elif order == Order.OVERLAP_OUTPUT_TO_TARGET:
            iou_argsort_target = np.argsort(-ious, axis=0)  # sort that each output counts to it's largest overlap target
        elif order == Order.OVERLAP_TARGET_TO_OUTPUT:
            iou_argsort_output = np.argsort(-ious, axis=1)  # sort that each target counts to it's largest overlap output
        elif order == Order.OVERLAP:
            iou_argsort = np.argsort(-ious.reshape(-1))

        targets_found = set()  # indices of found targets
        outputs_found = set()  # indices of found outputs

        for output_i in range(num_output):
            for target_i in range(num_target):
                # Get target t and output o indices
                if order == Order.CONFIDENCE_SCORE:
                    t = target_i
                    o = confidence_scores_argsort[output_i]  # output_max_i
                elif order == Order.OVERLAP_OUTPUT_TO_TARGET:
                    t = iou_argsort_target[target_i, output_i]  # target_max_i
                    o = output_i
                elif order == Order.OVERLAP_TARGET_TO_OUTPUT:
                    t = target_i
                    o = iou_argsort_output[target_i, output_i]
                elif order == Order.OVERLAP:
                    i = target_i * num_output + output_i
                    t = int(iou_argsort[i] / num_output)
                    o = iou_argsort[i] % num_output

                # to count as correct detection area of overlap must exceed the threshold
                if ious[t, o] > threshold and t not in targets_found and o not in outputs_found:
                    targets_found.add(t)
                    outputs_found.add(o)
                    if order == Order.CONFIDENCE_SCORE:
                        s = confidence_scores[confidence_scores_argsort[output_i]]
                    elif order == Order.OVERLAP_OUTPUT_TO_TARGET or order == Order.OVERLAP_TARGET_TO_OUTPUT or order == Order.OVERLAP:
                        s = 1.0
                    # TP
                    y_score += [to_one_vs_all(classes, class_id, s)]
                    y_true += [to_one_vs_all(classes, class_id, 1.0)]
                    break  # found output for target

        # all indices
        targets_all = set(range(num_target))
        outputs_all = set(range(num_output))

        targets_not_matching = targets_all - targets_found  # targets with no matching detection
        # FN
        y_score += [[0.0 for _ in range(num_classes)] for _ in targets_not_matching]
        y_true += [to_one_vs_all(classes, class_id, 1.0) for _ in targets_not_matching]

        # FP
        outputs_not_matching = outputs_all - outputs_found  # predictions with not matching targets
        if order == Order.CONFIDENCE_SCORE:
            y_score += [to_one_vs_all(classes, class_id, confidence_scores[i]) for i in outputs_not_matching]
        elif order == Order.OVERLAP_OUTPUT_TO_TARGET or order == Order.OVERLAP_TARGET_TO_OUTPUT or order == Order.OVERLAP:
            y_score += [to_one_vs_all(classes, class_id, 1.0) for _ in outputs_not_matching]
        y_true += [[0.0 for _ in range(num_classes)] for _ in outputs_not_matching]

    if len(classes) == 1:  # binary has different output shape
        y_score = [i[0] for i in y_score]
        y_true = [i[0] for i in y_true]

    return y_score, y_true


class GeometricMetrics(OnlineMetric):
    def __call__(self, output: list, target: list):
        results = [self.func(c1, c2) for c1, c2 in zip(output, target)]
        return results

    @classmethod
    def __repr__(cls):
        return cls.__name__


class APSklearn(GeometricMetrics):
    """
    Calculates the AP score with the function from Sklearn.
    Usage:  ap = APSklearn()
            ap(output_batch: list, target_batch: list) # list with cuboids OR true binary labels
    Cuboid parameters (dict):Format:'c' -> center of cuboid array[x,y, ... n]
                                    'd' -> dimension of cuboid array[length,width, ... n]
                                    'r' -> rotation of cuboid as 3x3 rot matrix
                                    'class_id' -> class id of cuboid (class id 0 is always background, do not use it)
                                    'confidence_score' -> confidence score of cuboid (must only be set if order=Order.CONFIDENCE_SCORE)
    Attributes:
        func (TYPE): function used in parent class

    Args:
        classes (list) : all class ids without 0 (0 ist always the background class)
        find_corres (bool): if True, input are cuboids and correspondences are calculated,
                            if False, input are true binary labels (one class) or binary label indicators. (multiple classes)
        threshold (float): threshold for the IOU, only overlaps of IOU > threshold are counted as correct
        order (Order): specifies which targets/outputs are matched first (see Order enum declaration for more infos)
    """

    def __init__(self, classes: list, find_corres=True, threshold=0.5, order=Order.CONFIDENCE_SCORE):
        self.func = self.average_precision_sklearn
        self.classes = classes
        self.find_corres = find_corres
        self.threshold = threshold
        self.order = order

    def average_precision_sklearn(self, output: list, target: list):
        if self.find_corres:
            y_score, y_true = find_correspondences(output, target, self.classes, self.threshold, self.order)
            y_score = np.array(y_score)
            y_true = np.array(y_true)
        else:
            y_score = output
            y_true = target

        return sklearn.metrics.average_precision_score(y_true, y_score)


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
                r=Rotation.from_euler('xyz', [45, 10, 30], degrees=True).as_dcm(),
                class_id=1),
           dict(c=np.array([0., 1., 2.]),
                d=np.array([4., 8., 10.]),
                r=Rotation.from_euler('xyz', [45, 10, 30], degrees=True).as_dcm(),
                class_id=1)
           ]
    target = [dict(c=np.array([0., 1., 2.]),
                   d=np.array([4., 8., 10.]),
                   r=Rotation.from_euler('xyz', [45, 10, 30], degrees=True).as_dcm(),
                   class_id=1),
              dict(c=np.array([0., 1.5, 2.]),
                   d=np.array([8., 8., 10.]),
                   r=Rotation.from_euler('xyz', [45, 20, 30], degrees=True).as_dcm(),
                   class_id=1)
              ]
    # usage example
    m = MeanDistanceError()
    m.test = "nachtr"
    metrics = [m, MeanDistanceError(), IoU()]
    result = {repr(m): m(out, target) for m in metrics}
    print(result)

    # 1.0
    metrics = [APSklearn([1], order=Order.OVERLAP)]
    result = {repr(m): m([[out[0]]], [[target[0]]]) for m in metrics}
    print(result)
    y_true, y_score = find_correspondences([out[0]], [target[0]], [1], 0.5, Order.OVERLAP)
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_true, y_score)
    print(precision)
    print(recall)
    print(thresholds)


    # 0.583
    metrics = [APSklearn([1], order=Order.OVERLAP)]
    result = {repr(m): m([out], [target]) for m in metrics}
    print(result)
    y_true, y_score = find_correspondences(out, target, [1], 0.5, Order.OVERLAP)
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_true, y_score)
    print(precision)
    print(recall)
    print(thresholds)

    precision = sklearn.metrics.precision_score(y_true, y_score)
    recall = sklearn.metrics.recall_score(y_true, y_score)
    print(precision)
    print(recall)
