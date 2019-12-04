# -*- coding: utf-8 -*-
# @Author: Daniel Dold
# @Date:   2019-11-20 10:08:49
# @Last Modified by:   Daniel Dold
# @Last Modified time: 2019-11-20 17:15:55
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.sparse import csr_matrix
from scipy.spatial import HalfspaceIntersection, ConvexHull
from scipy.spatial.qhull import QhullError
import logging
import os
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


class GeometricMetrics(OnlineMetric):
    def __call__(self, output: list, target: list):
        results = [self.func(c1, c2) for c1, c2 in zip(output, target)]
        return results

    @classmethod
    def __repr__(cls):
        return cls.__name__


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
        intersect = intersection(output, target)
        union = np.prod(output['d']) + np.prod(target['d']) - intersect
        return intersect / union


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
