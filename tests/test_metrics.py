# -*- coding: utf-8 -*-
# @Author: Daniel Dold and Markus Käppeler
# @Date:   2019-11-20 15:42:51
# @Last Modified by:   Daniel
# @Last Modified time: 2020-08-11 11:55:02
import numpy as np
from scipy.spatial.transform import Rotation
from sklearn import preprocessing
from ummon.metrics.geometric_metrics import halfspace_representation, intersection, MeanIoU, iou, calc_binary_confusion_matrix, find_correspondences, Sort, BinaryIoU, BinaryAccuracy, BinaryPrecision, BinaryRecall
import pytest
import logging
import scipy
from packaging import version

if version.parse(scipy.__version__) < version.parse("1.4.0"):
    Rotation.as_matrix = Rotation.as_dcm
    Rotation.from_matrix = Rotation.from_dcm


logging.getLogger().setLevel(0)


class TestMetrics:
    """docstring for TestMetrics"""
    box1_3d = dict(c=np.array([0., 0., 0.]),
                   d=np.array([2., 3., 4.]),
                   r=Rotation.from_euler('xyz', [0, 0, 0], degrees=True).as_matrix())
    box2_3d = dict(c=np.array([0., 0., 0.]),
                   d=np.array([4., 4., 4.]),
                   r=Rotation.from_euler('xyz', [0, 0, 0]).as_matrix())

    yaw1 = 0.
    yaw2 = 0.
    box1_2d = dict(c=np.array([0., 0.]),
                   d=np.array([2., 4.]),
                   r=np.array([[np.cos(yaw1), -np.sin(yaw1)],
                               [np.sin(yaw1), np.cos(yaw1)]]))
    box2_2d = dict(c=np.array([0., 0.]),
                   d=np.array([4., 4.]),
                   r=np.array([[np.cos(yaw2), -np.sin(yaw2)],
                               [np.sin(yaw2), np.cos(yaw2)]]))

    def test_halfspace_representation_2d(self):
        box = self.box1_2d.copy()
        halfspace = halfspace_representation(box)
        p1 = np.array([-1., -2., 1.])
        p2 = np.array([1., 2., 1.])
        assert np.isclose(np.dot(halfspace[0], p1), 0.).all()
        assert np.isclose(np.dot(halfspace[1], p1), 0.).all()
        assert np.isclose(np.dot(halfspace[2], p2), 0.).all()
        assert np.isclose(np.dot(halfspace[3], p2), 0.).all()

    def test_halfspace_representation_2d_translation(self):
        # with translation
        box = self.box1_2d.copy()
        box['c'] = np.array([5., -5])
        halfspace = halfspace_representation(box)
        p1 = np.array([4., -7., 1.])
        p2 = np.array([6., -3., 1.])
        assert np.isclose(np.dot(halfspace[0], p1), 0.).all()
        assert np.isclose(np.dot(halfspace[1], p1), 0.).all()
        assert np.isclose(np.dot(halfspace[2], p2), 0.).all()
        assert np.isclose(np.dot(halfspace[3], p2), 0.).all()

    def test_halfspace_representation_2d_rotation(self):
        # with rotation
        box = self.box1_2d.copy()
        yaw1 = np.pi / 4
        box['r'] = np.array([[np.cos(yaw1), -np.sin(yaw1)],
                             [np.sin(yaw1), np.cos(yaw1)]])
        box['c'] = np.array([10., -10])
        box['d'] = np.array([np.sqrt(8), np.sqrt(8)])
        halfspace = halfspace_representation(box)
        p1 = np.array([10., -12., 1.])
        p2 = np.array([10., -8., 1.])
        print(halfspace)
        assert np.isclose(np.dot(halfspace[0], p1), 0.).all()
        assert np.isclose(np.dot(halfspace[1], p1), 0.).all()
        assert np.isclose(np.dot(halfspace[2], p2), 0.).all()
        assert np.isclose(np.dot(halfspace[3], p2), 0.).all()

    def test_halfspace_representation_3d(self):
        # prepare
        box = self.box1_3d.copy()
        halfspace = halfspace_representation(box)
        p1 = np.array([-1., -1.5, -2., 1.])
        p2 = np.array([1., 1.5, 2., 1.])
        assert np.isclose(np.dot(halfspace[0], p1), 0)
        assert np.isclose(np.dot(halfspace[1], p1), 0)
        assert np.isclose(np.dot(halfspace[2], p1), 0)
        assert np.isclose(np.dot(halfspace[3], p2), 0)
        assert np.isclose(np.dot(halfspace[4], p2), 0)
        assert np.isclose(np.dot(halfspace[5], p2), 0)

    def test_halfspace_representation_3d_trans_rot(self):
        # prepare
        box = dict(c=np.array([10., -10., 0.]),
                   d=np.array([np.sqrt(8), np.sqrt(8), 4]),  # 2, 2, 4
                   r=Rotation.from_euler('xyz', [0, 0, 45], degrees=True).as_matrix())
        halfspace = halfspace_representation(box)
        p1 = np.array([10., -12, -2., 1.])
        p2 = np.array([10., -8, 2., 1.])
        assert np.isclose(np.dot(halfspace[0], p1), 0)
        assert np.isclose(np.dot(halfspace[1], p1), 0)
        assert np.isclose(np.dot(halfspace[2], p1), 0)
        assert np.isclose(np.dot(halfspace[3], p2), 0)
        assert np.isclose(np.dot(halfspace[4], p2), 0)
        assert np.isclose(np.dot(halfspace[5], p2), 0)

    def test_intersection(self):
        box1 = self.box1_3d.copy()
        box2 = self.box2_3d.copy()

        expected_res = 24.
        assert np.isclose(intersection(box1, box2), expected_res)

    def test_intersection_trans(self):
        box1 = dict(c=np.array([-1., -1., -1.]),
                    d=np.array([2, 4, 8.]),
                    r=Rotation.from_euler('xyz', [0, 0, 0], degrees=True).as_matrix())
        box2 = dict(c=np.array([1., 1., 1.]),
                    d=np.array([8., 4., 2.]),
                    r=Rotation.from_euler('xyz', [0, 0, 0]).as_matrix())

        expected_res = 8.
        assert np.isclose(intersection(box1, box2), expected_res)

    def test_intersection_trans_rot(self):
        box1 = dict(c=np.array([10., 10., 10.]),
                    d=np.array([np.sqrt(18), np.sqrt(18), 4]),
                    r=Rotation.from_euler('xyz', [0, 0, 45], degrees=True).as_matrix())
        box2 = dict(c=np.array([10., 10., 10.]),
                    d=np.array([4., 4., 4.]),
                    r=Rotation.from_euler('xyz', [0, 0, 0]).as_matrix())

        expected_res = (12 + 2) * 4
        assert np.isclose(intersection(box1, box2), expected_res)

    def test_rotation_as_quat_or_matrix(self):
        box1 = dict(c=np.array([10., 10., 10.]),
                    d=np.array([np.sqrt(18), np.sqrt(18), 4]),
                    r=Rotation.from_euler('xyz', [0, 0, 45], degrees=True).as_matrix())
        box11 = dict(c=np.array([10., 10., 10.]),
                     d=np.array([np.sqrt(18), np.sqrt(18), 4]),
                     r=Rotation.from_euler('xyz', [0, 0, 45], degrees=True).as_quat())
        box2 = dict(c=np.array([10., 10., 10.]),
                    d=np.array([4., 4., 4.]),
                    r=Rotation.from_euler('xyz', [0, 0, 0]).as_matrix())
        box22 = dict(c=np.array([10., 10., 10.]),
                     d=np.array([4., 4., 4.]),
                     r=Rotation.from_euler('xyz', [0, 0, 0]).as_quat())

        assert np.isclose(intersection(box1, box2), intersection(box11, box22))

    # targets
    box11 = dict(c=np.array([1., 0., 0.]),
                 d=np.array([2, 2, 2]),
                 r=Rotation.from_euler(
                     'xyz', [0, 0, 0], degrees=True).as_matrix(),
                 class_id=1)
    box12 = dict(c=np.array([1.2, 0.5, 0.]),
                 d=np.array([2, 2, 2]),
                 r=Rotation.from_euler('xyz', [0, 0, 0]).as_matrix(),
                 class_id=1)

    # outputs
    box13 = dict(c=np.array([0.5, 0., 0.]),
                 d=np.array([2, 2, 2]),
                 r=Rotation.from_euler(
                     'xyz', [0, 0, 0], degrees=True).as_matrix(),
                 class_id=1,
                 confidence_score=0.8)
    box14 = dict(c=np.array([1.2, 0., 0.]),
                 d=np.array([2, 2, 2]),
                 r=Rotation.from_euler('xyz', [0, 0, 0]).as_matrix(),
                 class_id=1,
                 confidence_score=0.6)

    def test_check_data(self):
        assert iou(self.box11, self.box13) > 0.5
        assert iou(self.box12, self.box13) < 0.5
        assert iou(self.box11, self.box14) > 0.5
        assert iou(self.box12, self.box14) > 0.5

        assert iou(self.box11, self.box13) < iou(self.box11, self.box14)
        assert self.box13['confidence_score'] > self.box14['confidence_score']

    def test_find_correspondences_confidence_score_binary_0(self):
        output_to_target, target_to_output = find_correspondences(output=[self.box13, self.box14],
                                                                  target=[
                                                                      self.box11, self.box12], threshold=0.5,
                                                                  sort=Sort.CONFIDENCE_SCORE)
        output_to_target == np.array([0, 1])
        target_to_output == np.array([0, 1])

    def test_find_correspondences_iou_binary_0(self):
        output_to_target, target_to_output = find_correspondences(output=[self.box14, self.box13],
                                                                  target=[
                                                                      self.box11, self.box12], threshold=0.5,
                                                                  sort=Sort.IOU)
        output_to_target == np.array([0, -1])
        target_to_output == np.array([0, -1])

    def test_find_correspondences_iou_binary_1(self):
        output_to_target, target_to_output = find_correspondences(output=[self.box13, self.box14],
                                                                  target=[
                                                                      self.box11, self.box12], threshold=0.5,
                                                                  sort=Sort.IOU)
        output_to_target == np.array([-1, 0])
        target_to_output == np.array([1, -1])

    box1 = dict(c=np.array([10., 10., 10.]),
                d=np.array([5, 5, 5]),
                r=Rotation.from_euler('xyz', [0, 0, 0]).as_matrix(),
                class_id=1)
    box2 = dict(c=np.array([10., 10., 10.]),
                d=np.array([5., 5., 5.]),
                r=Rotation.from_euler('xyz', [0, 0, 0]).as_matrix())

    box3 = dict(c=np.array([20., 20., 20.]),
                d=np.array([5., 5., 5.]),
                r=Rotation.from_euler('xyz', [0, 0, 0]).as_matrix(),
                class_id=1)
    box4 = dict(c=np.array([21., 20., 20.]),
                d=np.array([5., 5., 5.]),
                r=Rotation.from_euler('xyz', [0, 0, 0]).as_matrix(),)

    box5 = dict(c=np.array([30., 30., 30.]),
                d=np.array([5., 5., 5.]),
                r=Rotation.from_euler('xyz', [0, 0, 0]).as_matrix(),
                class_id=1)

    box6 = dict(c=np.array([40., 40., 40.]),
                d=np.array([5., 5., 5.]),
                r=Rotation.from_euler('xyz', [0, 0, 0]).as_matrix())

    box7 = dict(c=np.array([50., 50., 50.]),
                d=np.array([5., 5., 5.]),
                r=Rotation.from_euler('xyz', [0, 0, 0]).as_matrix())

    box8 = dict(c=np.array([60., 60., 60.]),
                d=np.array([5., 5., 5.]),
                r=Rotation.from_euler('xyz', [0, 0, 0]).as_matrix(),
                class_id=0)

    box9 = dict(c=np.array([70., 70., 70.]),
                d=np.array([5., 5., 5.]),
                r=Rotation.from_euler('xyz', [0, 0, 0]).as_matrix(),
                class_id=0)

    box10 = dict(c=np.array([70.5, 70.5, 70.5]),
                 d=np.array([5., 5., 5.]),
                 r=Rotation.from_euler('xyz', [0, 0, 0]).as_matrix())

    def test_calc_binary_confusion_matrix_0(self):
        TP, FP, FN_0, FN_1, TN = calc_binary_confusion_matrix(output=[[self.box1, self.box3, self.box5]],
                                                              target=[[self.box2, self.box4]])
        assert TP == 2
        assert FP == 1
        assert FN_0 == 0
        assert FN_1 == 0
        assert TN == 0

    def test_calc_binary_confusion_matrix_1(self):
        TP, FP, FN_0, FN_1, TN = calc_binary_confusion_matrix(output=[[self.box1, self.box3, self.box5]],
                                                              target=[[self.box2, self.box4, self.box6, self.box7]])
        assert TP == 2
        assert FP == 1
        assert FN_0 == 0
        assert FN_1 == 2
        assert TN == 0

    def test_calc_binary_confusion_matrix_2(self):
        TP, FP, FN_0, FN_1, TN = calc_binary_confusion_matrix(output=[[self.box1, self.box3, self.box5, self.box8, self.box9]],
                                                              target=[[self.box2, self.box4, self.box6, self.box7, self.box10]])
        assert TP == 2
        assert FP == 1
        assert FN_0 == 1
        assert FN_1 == 2
        assert TN == 1

    def test_binary_iou(self):
        binary_iou = BinaryIoU()
        assert binary_iou([[self.box1]], [[self.box2]]) == 1
        assert binary_iou([[self.box1, self.box3, self.box5, self.box8, self.box9]],
                          [[self.box2, self.box4, self.box6, self.box7, self.box10]]) == 2 / (2 + 1 + 1 + 2)

    def test_binary_accuracy(self):
        binary_accuracy = BinaryAccuracy()
        assert binary_accuracy([[self.box1]], [[self.box2]]) == 1
        assert binary_accuracy([[self.box1, self.box3, self.box5, self.box8, self.box9]],
                               [[self.box2, self.box4, self.box6, self.box7, self.box10]]) == (2 + 1) / (2 + 1 + 1 + 2 + 1)

    def test_binary_precision(self):
        binary_precision = BinaryPrecision()
        assert binary_precision([[self.box1]], [[self.box2]]) == 1
        assert binary_precision([[self.box1, self.box3, self.box5, self.box8, self.box9]],
                                [[self.box2, self.box4, self.box6, self.box7, self.box10]]) == 2 / (2 + 1)

    def test_binary_recall(self):
        binary_recall = BinaryRecall()
        assert binary_recall([[self.box1]], [[self.box2]]) == 1
        assert binary_recall([[self.box1, self.box3, self.box5, self.box8, self.box9]],
                             [[self.box2, self.box4, self.box6, self.box7, self.box10]]) == 2 / (2 + 1 + 2)

    def test_intersection_over_union(self):
        # prepare
        iou = MeanIoU()
        box1 = [dict(c=np.array([0., 1., 4.]),
                     d=np.array([4., 8., 10.]),
                     r=Rotation.from_euler('xyz', [0, 0, 30], degrees=True).as_matrix()),
                dict(c=np.array([0., 1., 4.]),
                     d=np.array([4., 8., 10.]),
                     r=Rotation.from_euler('xyz', [0, 0, 30], degrees=True).as_matrix()),
                dict(c=np.array([0., 1., 4.]),
                     d=np.array([4., 8., 10.]),
                     r=Rotation.from_euler('xyz', [0, 0, 30], degrees=True).as_matrix())]
        box2 = [dict(c=np.array([0., 1., 2]),
                     d=np.array([4., 8., 10.]),
                     r=Rotation.from_euler('xyz', [0, 0, 30], degrees=True).as_matrix()),
                dict(c=np.array([0., 1., 4.]),
                     d=np.array([4., 8., 10.]),
                     r=Rotation.from_euler('xyz', [0, 0, 30], degrees=True).as_matrix()),
                dict(c=np.array([20., 1., 4.]),
                     d=np.array([4., 8., 10.]),
                     r=Rotation.from_euler('xyz', [45, 45, 30], degrees=True).as_matrix())]
        # test
        result = iou([box1], [box2])
        expected_res = np.mean([4 / 6, 1, 0]), np.var([4 / 6, 1, 0])
        assert np.isclose(result, expected_res).all()
