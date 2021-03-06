# -*- coding: utf-8 -*-
# @Author: Daniel Dold, Markus Käppeler
# @Date:   2019-11-20 10:08:49
# @Last Modified by:   Daniel
# @Last Modified time: 2020-08-12 13:18:57
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.sparse import csr_matrix
from scipy.spatial import HalfspaceIntersection, ConvexHull
from scipy.spatial.qhull import QhullError
import logging
import os
from sklearn.metrics import average_precision_score, precision_recall_curve
from .base import *
from enum import Enum
import itertools
import scipy
from packaging import version
from ummon.utils.average_utils import OnlineAverage
import itertools

if version.parse(scipy.__version__) < version.parse("1.4.0"):
    Rotation.as_matrix = Rotation.as_dcm
    Rotation.from_matrix = Rotation.from_dcm

__all__ = ['MeanIoU', 'MeanDistanceError', 'BinaryAccuracy', 'BinaryIoU', 'BinaryF1', 'BinaryRecall', 'BinaryPrecision',
           'AveragePrecision']

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
                            'r' -> rotation of cuboid as 3x3 rot matrix or quaternion

    Returns:
        np.array: cuboid in half space representation (shape [4xn+1])
    """
    # create normal vectors from a sparse matrix
    if len(cuboid['r'].shape) == 1:
        cub_rot_m = Rotation.from_quat(cuboid['r']).as_matrix()
    else:
        cub_rot_m = cuboid['r']
    p_dim = cuboid['d'].shape[0]
    row = np.arange(2 * p_dim)
    col = np.array(list(np.arange(p_dim)) * 2)
    data = np.array([-1, 1]).repeat(p_dim)
    norm_vec = csr_matrix((data, (row, col)), shape=(
        p_dim * 2, p_dim)).toarray()  # 4x2 or 6x3
    # Rotation of axis aligned normal vectors
    norm_vec_rot = np.matmul(norm_vec, cub_rot_m.T)  # 4x2 or 6x3

    # compute d parameter of plane representation
    # p1 and p2 span the bounding cuboid volume (diagonal to each other)
    p1 = cuboid['c'] + np.matmul(-(cuboid['d'] / 2), cub_rot_m.T)
    p2 = cuboid['c'] + np.matmul((cuboid['d'] / 2), cub_rot_m.T)
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


class Sort(Enum):
    IOU = 1  # Sort by IOU
    CONFIDENCE_SCORE = 2  # Sort by output confidence score
    DISTANCE = 3


def find_correspondences(output: list, target: list, threshold: float, sort: Sort) -> tuple:
    """
    Finds correspondences between outputs and targets cuboids by matching cuboids which exceed the given IOU threshold.
    Args:
        output (list): list of predicted cuboids (dict)
        target (list): list of target cuboids (dict)
        threshold (float): threshold for IOU to count as correct detection
        sort (Sort enum): sort to match output and targets, if set to Sort.CONFIDENCE_SCORE, output cuboids must
            contain confidence_score.

    Returns:
        tuple (2,): tuple of output_to_target and target_to_ouput. Each is a np array and maps indices
            from output to target and the other way round for each correspondence.
            Array contains 'inf' if there is no match for the output or target.

    """
    num_output = len(output)
    num_target = len(target)

    if sort == Sort.DISTANCE:
        similarity_measure = MeanDistanceError().mean_distance_error
    else:
        similarity_measure = iou

    # calculate all ious
    sim_m = np.zeros((num_target, num_output))
    for i_target, cuboid_target in enumerate(target):
        for i_output, cuboid_output in enumerate(output):
            sim_m[i_target, i_output] = similarity_measure(
                cuboid_output, cuboid_target)

    # calculate sorting, which to match first
    if sort == Sort.CONFIDENCE_SCORE:
        confidence_scores = np.array(
            [cuboid['confidence_score'] for cuboid in output])
        confidence_scores_argsort = np.argsort(-confidence_scores)
    elif sort == Sort.IOU:
        iou_argsort = np.argsort(-sim_m.reshape(-1))
    elif sort == Sort.DISTANCE:
        iou_argsort = np.argsort(sim_m.reshape(-1))
    else:
        raise NotImplemented

    # maps indices from output to target and the other way round for each correspondence
    output_to_target = np.full((num_output,), float('inf'))
    target_to_output = np.full((num_target,), float('inf'))

    for i, (output_i, target_i) in enumerate(itertools.product(list(range(num_output)), list(range(num_target)))):
        # Get target t and output o indices
        if sort == Sort.CONFIDENCE_SCORE:
            t = target_i
            o = confidence_scores_argsort[output_i]
        elif sort == Sort.IOU or sort == Sort.DISTANCE:
            t = int(iou_argsort[i] / num_output)
            o = iou_argsort[i] % num_output

        # to count as correct detection area the iou must exceed the threshold
        if sim_m[t, o] > threshold and output_to_target[o] == float('inf') and target_to_output[t] == float('inf'):
            output_to_target[o] = t
            target_to_output[t] = o

    return output_to_target, target_to_output


def calc_binary_confusion_matrix(output: list, target: list):
    """
    Calculates TP, FP, FN_0, FN_1, TN for a output and target cuboid list of list (multiple scenes) by finding
    correspondences between output and target cuboids with a IOU > 0.5. Correspondences with high IOU are matched first.
    FN_0 are false negatives where a ground truth cuboid machted, but prediction has class id 0.
    FN_1 are false negatives where no ground truth cuboid machted.
    Args:
        output (list): list of predicted cuboids (dict)
        target (list): list of target cuboids (dict)

    Returns:
        tuple: number of TP, FP, FN_0, FN_1, TN

    """

    TP = FP = FN_0 = FN_1 = TN = 0
    for o, t in zip(output, target):  # iter over scenes
        output_to_target, target_to_output = find_correspondences(
            o, t, 0.5, Sort.IOU)

        output_class_ids = np.array([cuboid['class_id']
                                     for cuboid in o], dtype=bool)

        assert (~np.isinf(output_to_target)).sum() == (
            ~np.isinf(target_to_output)).sum()
        TP += np.logical_and(~(np.isinf(output_to_target)),
                             output_class_ids).sum()
        FP += np.logical_and(np.isinf(output_to_target),
                             output_class_ids).sum()
        FN_0 += np.logical_and(~(np.isinf(output_to_target)),
                               np.logical_not(output_class_ids)).sum()
        FN_1 += (np.isinf(target_to_output)).sum()
        TN += np.logical_and(np.isinf(output_to_target),
                             np.logical_not(output_class_ids)).sum()
    return TP, FP, FN_0, FN_1, TN


class ObjectDetectionMetric(OfflineMetric):
    def __call__(self, output: list, target: list):
        return self.func(output, target)

    @classmethod
    def __repr__(cls):
        return cls.__name__


class BinaryIoU(ObjectDetectionMetric):
    """Compute IoU with confusion matrix (TP / (TP + FP + FN)) for binary object detection task.
    Usage:  binary_iou = BinaryIoU()
            binary_iou(cuboids_output: list of list, cuboids_target: list of list) # cuboids wrapped in list of list (scenes)
    Cuboid parameters (dict):Format:'c' -> center of cuboid array[x,y, ... n]
                                    'd' -> dimension of cuboid array[length,width, ... n]
                                    'r' -> rotation of cuboid as 3x3 rot matrix
                                    'class_id' -> class_id, either 0 or 1
    Attributes:
        func (TYPE): function used in parent class
    """

    def __init__(self):
        self.func = self.accuracy

    @staticmethod
    def accuracy(output: list, target: list):
        TP, FP, FN_0, FN_1, TN = calc_binary_confusion_matrix(output, target)
        return TP / (TP + FP + FN_0 + FN_1)


class BinaryAccuracy(ObjectDetectionMetric):
    """Compute Accuracy with confusion matrix ((TP + TN) / (TP + TN + FP + FN + FN)) for binary object detection task.
    Usage:  binary_accuracy = BinaryAccuracy()
            binary_accuracy(cuboids_output: list of list, cuboids_target: list of list) # cuboids wrapped in list of list (scenes)
    Cuboid parameters (dict):Format:'c' -> center of cuboid array[x,y, ... n]
                                    'd' -> dimension of cuboid array[length,width, ... n]
                                    'r' -> rotation of cuboid as 3x3 rot matrix
                                    'class_id' -> class_id, either 0 or 1
    Attributes:
        func (TYPE): function used in parent class
    """

    def __init__(self):
        self.func = self.accuracy

    @staticmethod
    def accuracy(output: list, target: list):
        TP, FP, FN_0, FN_1, TN = calc_binary_confusion_matrix(output, target)
        return (TP + TN) / (TP + TN + FP + FN_0 + FN_1)


class BinaryPrecision(ObjectDetectionMetric):
    """Compute Precision with confusion matrix (TP / (TP + FP)) for binary object detection task.
    Usage:  binary_precision = BinaryPrecision()
            binary_precision(cuboids_output: list of list, cuboids_target: list of list) # cuboids wrapped in list of list (scenes)
    Cuboid parameters (dict):Format:'c' -> center of cuboid array[x,y, ... n]
                                    'd' -> dimension of cuboid array[length,width, ... n]
                                    'r' -> rotation of cuboid as 3x3 rot matrix
                                    'class_id' -> class_id, either 0 or 1
    Attributes:
        func (TYPE): function used in parent class
    """

    def __init__(self):
        self.func = self.precision

    @staticmethod
    def precision(output: list, target: list):
        TP, FP, FN_0, FN_1, TN = calc_binary_confusion_matrix(output, target)
        return TP / (TP + FP)


class BinaryRecall(ObjectDetectionMetric):
    """Compute Recall with confusion matrix (TP / (TP + FN)) for binary object detection task.
    Usage:  binary_recall = BinaryRecall()
            binary_recall(cuboids_output: list of list, cuboids_target: list of list) # cuboids wrapped in list of list (scenes)
    Cuboid parameters (dict):Format:'c' -> center of cuboid array[x,y, ... n]
                                    'd' -> dimension of cuboid array[length,width, ... n]
                                    'r' -> rotation of cuboid as 3x3 rot matrix
                                    'class_id' -> class_id, either 0 or 1
    Attributes:
        func (TYPE): function used in parent class
    """

    def __init__(self):
        self.func = self.precision

    @staticmethod
    def precision(output: list, target: list):
        TP, FP, FN_0, FN_1, TN = calc_binary_confusion_matrix(output, target)
        return TP / (TP + FN_0 + FN_1)


class BinaryF1(ObjectDetectionMetric):
    """Compute F1 with confusion matrix (2 * (precision * recall) / (precision + recall)) for binary object detection task.
    Usage:  binary_f1 = BinaryF1()
            binary_f1(cuboids_output: list of list, cuboids_target: list of list) # cuboids wrapped in list of list (scenes)
    Cuboid parameters (dict):Format:'c' -> center of cuboid array[x,y, ... n]
                                    'd' -> dimension of cuboid array[length,width, ... n]
                                    'r' -> rotation of cuboid as 3x3 rot matrix
                                    'class_id' -> class_id, either 0 or 1
    Attributes:
        func (TYPE): function used in parent class
    """

    def __init__(self):
        self.func = self.precision

    @staticmethod
    def precision(output: list, target: list):
        TP, FP, FN_0, FN_1, TN = calc_binary_confusion_matrix(output, target)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN_0 + FN_1)
        return 2 * (precision * recall) / (precision + recall)


class GeometricMetrics(OnlineMetric):
    def __init__(self,
                 find_correspondences__th=-1.0,
                 find_correspondences__sort: Sort=Sort.DISTANCE):
        self.threshold_ = find_correspondences__th
        self.sort_ = find_correspondences__sort
        self.avg_f = OnlineAverage()
        self.avg_f2 = OnlineAverage()
        self.avg_f.reset()
        self.avg_f2.reset()

    def __call__(self, output: list, target: list):
        results = [self.__metric_per_item(c1, c2)
                   for c1, c2 in zip(output, target)]
        mean = self.avg_f(list(itertools.chain(*results)))
        mean2 = self.avg_f2(np.power(list(itertools.chain(*results)), 2))
        var = mean2 - mean**2
        return mean, var

    def reset(self):
        self.avg_f.reset()
        self.avg_f2.reset()

    def __metric_per_item(self, output_i, target_i):  # item out of minibatch
        o_to_t, t_to_o = find_correspondences(output_i,
                                              target_i,
                                              threshold=self.threshold_,
                                              sort=self.sort_)
        # ignore targets without a corresponding prediction
        mask = ~np.isinf(t_to_o)
        sort = t_to_o[mask].astype(np.int)
        output_i = np.array(output_i)[sort]
        target_i = np.array(target_i)[mask]
        # compute metric per item
        return [self.func(o, t) for o, t in zip(output_i, target_i)]

    @classmethod
    def __repr__(cls):
        return cls.__name__


class MeanIoU(GeometricMetrics):
    """Compute the Intersection over Union (IoU) of two arbitrary 2d-/ 3d cuboids
    Usage:  iou = IoU()
            iou(cuboids_1: list, cuboids_2: list)  # cuboids wrapped in list
    Cuboid parameters (dict):Format:'c' -> center of cuboid array[x,y, ... n]
                                    'd' -> dimension of cuboid array[length,width, ... n]
                                    'r' -> rotation of cuboid as 3x3 rot matrix
    Attributes:
        func (TYPE): function used in parent class
    """

    def __init__(self, **kwargs):
        self.func = self.intersection_over_union
        super().__init__(**kwargs)

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

    def __init__(self, **kwargs):
        self.func = self.mean_distance_error
        super().__init__(**kwargs)

    @staticmethod
    def mean_distance_error(output: dict, target: dict):
        error_vec = output['c'] - target['c']
        error = np.sqrt(np.dot(error_vec, error_vec))
        return error


class MeanDimensionError(GeometricMetrics):
    """Compute the geometric distance of two cuboids
    Usage:  dist = MeanDistanceError()
            dist(cuboids_1: list, cuboids_2: list) # cuboids wrapped in list
    Cuboid parameters (dict):Format:'c' -> center of cuboid array[x,y, ... n]
                                    'd' -> dimension of cuboid array[length,width, ... n]
                                    'r' -> rotation of cuboid as 3x3 rot matrix
    Attributes:
        func (TYPE): function used in parent class
    """

    def __init__(self,
                 dimension,
                 target_dimension_order=np.array([0, 1, 2]),
                 **kwargs):
        self.func = self.dimension_error
        self.dimension_ = dimension
        self.td_order_ = target_dimension_order
        super().__init__(**kwargs)

    def dimension_error(self, output: dict, target: dict):
        error = output['d'][self.dimension_] - \
            target['d'][self.td_order_][self.dimension_]
        return np.abs(error)

    def __repr__(self):
        return self.__class__.__name__ + "_dim_{}".format(self.dimension_)


class AveragePrecision(ObjectDetectionMetric):
    """Compute the Average Precision of two lists arbitrary 2d-/ 3d cuboids
       Usage:  ap = AveragePrecision()
               ap(cuboids_1: list, cuboids_2: list)  # cuboids wrapped in list
       Cuboid parameters (dict):Format:'c' -> center of cuboid array[x,y, ... n]
                                       'd' -> dimension of cuboid array[length,width, ... n]
                                       'r' -> rotation of cuboid as 3x3 rot matrix
                                       'confidence_score' -> confidence_score of bbox
       Attributes:
           func (TYPE): function used in parent class
       """

    def __init__(self, return_prec_rec_curve=False, iou_th=0.5):
        self.func = self.calc_score
        self.return_prec_rec_curve = return_prec_rec_curve
        self.iou_th_ = iou_th

    def calc_score(self, output_list, targets_list):
        y_true_list = []
        scores_list = []
        for o, t in zip(output_list, targets_list):  # iter over single item
            output_to_target, target_to_output = find_correspondences(
                o, t, self.iou_th_, Sort.IOU)
            n_pred = len(output_to_target)
            n_fn = np.isinf(target_to_output).sum()
            y_true = np.ones(n_pred + n_fn)
            y_true[:n_pred][np.isinf(output_to_target)] = 0

            scores = np.zeros_like(y_true)
            scores[:n_pred] = [cuboid['confidence_score'] for cuboid in o]

            y_true_list.extend(y_true)
            scores_list.extend(scores)

        precision, recall, threshold = precision_recall_curve(
            y_true_list, scores_list)
        if threshold[0] == 0:
            average_precision = - \
                np.sum(np.diff(recall[1:]) * np.array(precision)[1:-1])
        else:
            average_precision = - \
                np.sum(np.diff(recall) * np.array(precision)[:-1])
        if self.return_prec_rec_curve:
            return average_precision, (precision, recall, threshold)
        return average_precision

    def __repr__(self):
        return self.__class__.__name__ + "_{}".format(self.iou_th_)


class DataCollector(OnlineMetric):
    """docstring for DataCollector"""
    output_list = []
    target_list = []

    @classmethod
    def __call__(cls, output, target):
        cls.output_list.extend(output)
        cls.target_list.extend(target)
        return cls

    @classmethod
    def reset(cls):
        cls.output_list = []
        cls.target_list = []


if __name__ == '__main__':
    # usage example
    # prepare demo data
    out = [dict(c=np.array([0., 1., 2.]),
                d=np.array([4., 8., 10.]),
                r=Rotation.from_euler(
                    'xyz', [45, 10, 30], degrees=True).as_matrix(),
                class_id=1),
           dict(c=np.array([0., 1., 2.]),
                d=np.array([4., 8., 10.]),
                r=Rotation.from_euler(
                    'xyz', [45, 10, 30], degrees=True).as_matrix(),
                class_id=1)
           ]
    target = [dict(c=np.array([0., 1., 2.]),
                   d=np.array([4., 8., 10.]),
                   r=Rotation.from_euler(
                       'xyz', [45, 10, 30], degrees=True).as_matrix(),
                   class_id=1),
              dict(c=np.array([0., 1.5, 2.]),
                   d=np.array([8., 8., 10.]),
                   r=Rotation.from_euler(
                       'xyz', [45, 20, 30], degrees=True).as_matrix(),
                   class_id=1)
              ]
    # usage example
    m = MeanDistanceError()
    m.test = "nachtr"
    metrics = [m, MeanDistanceError(), IoU()]
    result = {repr(m): m(out, target) for m in metrics}
    print(result)

    metrics = [BinaryIoU(), BinaryAccuracy()]
    result = {repr(m): m([out], [target]) for m in metrics}
    print(result)
