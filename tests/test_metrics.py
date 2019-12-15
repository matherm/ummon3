# -*- coding: utf-8 -*-
# @Author: Daniel Dold
# @Date:   2019-11-20 15:42:51
# @Last Modified by:   Daniel Dold
# @Last Modified time: 2019-11-20 16:12:35
import numpy as np
from scipy.spatial.transform import Rotation
from sklearn import preprocessing
from ummon.metrics.geometric_metrics import halfspace_representation, intersection, IoU, iou, find_correspondences, APSklearn, Order
import pytest
import logging

logging.getLogger().setLevel(0)


class TestMetrics:
    """docstring for TestMetrics"""
    box1_3d = dict(c=np.array([0., 0., 0.]),
                   d=np.array([2., 3., 4.]),
                   r=Rotation.from_euler('xyz', [0, 0, 0], degrees=True).as_dcm())
    box2_3d = dict(c=np.array([0., 0., 0.]),
                   d=np.array([4., 4., 4.]),
                   r=Rotation.from_euler('xyz', [0, 0, 0]).as_dcm())

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
                   r=Rotation.from_euler('xyz', [0, 0, 45], degrees=True).as_dcm())
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
                    r=Rotation.from_euler('xyz', [0, 0, 0], degrees=True).as_dcm())
        box2 = dict(c=np.array([1., 1., 1.]),
                    d=np.array([8., 4., 2.]),
                    r=Rotation.from_euler('xyz', [0, 0, 0]).as_dcm())

        expected_res = 8.
        assert np.isclose(intersection(box1, box2), expected_res)

    def test_intersection_trans_rot(self):
        box1 = dict(c=np.array([10., 10., 10.]),
                    d=np.array([np.sqrt(18), np.sqrt(18), 4]),
                    r=Rotation.from_euler('xyz', [0, 0, 45], degrees=True).as_dcm())
        box2 = dict(c=np.array([10., 10., 10.]),
                    d=np.array([4., 4., 4.]),
                    r=Rotation.from_euler('xyz', [0, 0, 0]).as_dcm())

        expected_res = (12 + 2) * 4
        assert np.isclose(intersection(box1, box2), expected_res)

    box1 = dict(c=np.array([10., 10., 10.]),
                d=np.array([5, 5, 5]),
                r=Rotation.from_euler('xyz', [0, 0, 0], degrees=True).as_dcm(),
                class_id=1)
    box2 = dict(c=np.array([10., 10., 10.]),
                d=np.array([5., 5., 5.]),
                r=Rotation.from_euler('xyz', [0, 0, 0]).as_dcm(),
                class_id=1)

    box3 = dict(c=np.array([20., 20., 20.]),
                d=np.array([5., 5., 5.]),
                r=Rotation.from_euler('xyz', [0, 0, 0], degrees=True).as_dcm(),
                class_id=1)
    box4 = dict(c=np.array([21., 20., 20.]),
                d=np.array([5., 5., 5.]),
                r=Rotation.from_euler('xyz', [0, 0, 0]).as_dcm(),
                class_id=1)

    box5 = dict(c=np.array([30., 30., 30.]),
                d=np.array([5., 5., 5.]),
                r=Rotation.from_euler('xyz', [0, 0, 0]).as_dcm(),
                class_id=1)

    box6 = dict(c=np.array([40., 40., 40.]),
                d=np.array([5., 5., 5.]),
                r=Rotation.from_euler('xyz', [0, 0, 0]).as_dcm(),
                class_id=1)

    box7 = dict(c=np.array([50., 50., 50.]),
                d=np.array([5., 5., 5.]),
                r=Rotation.from_euler('xyz', [0, 0, 0]).as_dcm(),
                class_id=1)

    def test_find_correspondences_overlap_binary_0(self):
        classes = [0, 1]
        y_score, y_true = find_correspondences(outputs=[self.box1, self.box3, self.box5], targets=[self.box2, self.box4], classes=classes, threshold=0.5, order=Order.OVERLAP)

        y_score, y_true = np.array(y_score), np.array(y_true)

        # from binary to multi class
        lb = preprocessing.LabelBinarizer()
        lb.fit(classes)
        y_score = lb.inverse_transform(y_score)
        y_true = lb.inverse_transform(y_true)

        y_score_true = np.stack((np.array(y_score), np.array(y_true)), axis=-1)

        assert len(np.where((y_score_true == (1, 1)).all(axis=1))[0]) == 2  # TP
        assert len(np.where((y_score_true == (1, 0)).all(axis=1))[0]) == 1  # FP

    def test_find_correspondences_overlap_binary_1(self):
        classes = [0, 1]
        y_score, y_true = find_correspondences(outputs=[self.box1, self.box3, self.box5],
                                               targets=[self.box2, self.box4, self.box6, self.box7], classes=classes,
                                               threshold=0.5, order=Order.OVERLAP)

        y_score, y_true = np.array(y_score), np.array(y_true)

        # from binary to multi class
        lb = preprocessing.LabelBinarizer()
        lb.fit(classes)
        y_score = lb.inverse_transform(y_score)
        y_true = lb.inverse_transform(y_true)

        y_score_true = np.stack((np.array(y_score), np.array(y_true)), axis=-1)

        assert len(np.where((y_score_true == (1, 1)).all(axis=1))[0]) == 2  # TP
        assert len(np.where((y_score_true == (1, 0)).all(axis=1))[0]) == 1  # FP
        assert len(np.where((y_score_true == (0, 1)).all(axis=1))[0]) == 2  # FN

    def test_find_correspondences_overlap_multiclass_0(self):
        box1 = self.box1.copy()
        box1['class_id'] = 2
        box2 = self.box2.copy()
        box2['class_id'] = 2

        box3 = self.box3.copy()
        box3['class_id'] = 3
        box4 = self.box4.copy()
        box4['class_id'] = 4

        classes = [0, 1, 2, 3, 4]
        y_score, y_true = find_correspondences(outputs=[box1, box3, self.box5],
                                               targets=[box2, box4, self.box6, self.box7], classes=classes,
                                               threshold=0.5, order=Order.OVERLAP)

        y_score, y_true = np.array(y_score), np.array(y_true)

        # from binary to multi class
        lb = preprocessing.LabelBinarizer()
        lb.fit(classes)
        y_score = lb.inverse_transform(y_score)
        y_true = lb.inverse_transform(y_true)

        y_score_true = np.stack((np.array(y_score), np.array(y_true)), axis=-1)

        assert len(np.where((y_score_true == (2, 2)).all(axis=1))[0]) == 1  # TP, 1+2
        assert len(np.where((y_score_true == (3, 0)).all(axis=1))[0]) == 1  # FP, 3
        assert len(np.where((y_score_true == (0, 4)).all(axis=1))[0]) == 1  # FN, 4
        assert len(np.where((y_score_true == (1, 0)).all(axis=1))[0]) == 1  # FP, 5
        assert len(np.where((y_score_true == (0, 1)).all(axis=1))[0]) == 2  # FN, 6+7

    # targets
    box11 = dict(c=np.array([1., 0., 0.]),
                d=np.array([2, 2, 2]),
                r=Rotation.from_euler('xyz', [0, 0, 0], degrees=True).as_dcm(),
                class_id=1)
    box12 = dict(c=np.array([1.2, 0.5, 0.]),
                d=np.array([2, 2, 2]),
                r=Rotation.from_euler('xyz', [0, 0, 0]).as_dcm(),
                class_id=1)

    # outputs
    box13 = dict(c=np.array([0.5, 0., 0.]),
                d=np.array([2, 2, 2]),
                r=Rotation.from_euler('xyz', [0, 0, 0], degrees=True).as_dcm(),
                class_id=1,
                confidence_score=0.8)
    box14 = dict(c=np.array([1.2, 0., 0.]),
                d=np.array([2, 2, 2]),
                r=Rotation.from_euler('xyz', [0, 0, 0]).as_dcm(),
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
        classes = [0, 1]
        y_score, y_true = find_correspondences(outputs=[self.box13, self.box14],
                                               targets=[self.box11, self.box12], classes=classes, threshold=0.5,
                                               order=Order.CONFIDENCE_SCORE)

        y_score, y_true = np.array(y_score), np.array(y_true)

        # from binary to multi class
        lb = preprocessing.LabelBinarizer()
        lb.fit(classes)
        y_score = lb.inverse_transform(y_score)
        y_true = lb.inverse_transform(y_true)

        y_score_true = np.stack((np.array(y_score), np.array(y_true)), axis=-1)

        assert len(np.where((y_score_true == (1, 1)).all(axis=1))[0]) == 2  # TP
        assert len(np.where((y_score_true == (1, 0)).all(axis=1))[0]) == 0  # FP
        assert len(np.where((y_score_true == (0, 1)).all(axis=1))[0]) == 0  # FN

    def test_find_correspondences_overlap_binary_2(self):
        classes = [0, 1]
        y_score, y_true = find_correspondences(outputs=[self.box13, self.box14],
                                               targets=[self.box11, self.box12], classes=classes, threshold=0.5,
                                               order=Order.OVERLAP_OUTPUT_TO_TARGET)

        y_score, y_true = np.array(y_score), np.array(y_true)

        # from binary to multi class
        lb = preprocessing.LabelBinarizer()
        lb.fit(classes)
        y_score = lb.inverse_transform(y_score)
        y_true = lb.inverse_transform(y_true)

        y_score_true = np.stack((np.array(y_score), np.array(y_true)), axis=-1)

        assert len(np.where((y_score_true == (1, 1)).all(axis=1))[0]) == 2  # TP
        assert len(np.where((y_score_true == (1, 0)).all(axis=1))[0]) == 0  # FP
        assert len(np.where((y_score_true == (0, 1)).all(axis=1))[0]) == 0  # FN

    def test_find_correspondences_overlap_binary_3(self):
        classes = [0, 1]
        y_score, y_true = find_correspondences(outputs=[self.box14, self.box13],
                                               targets=[self.box12, self.box11], classes=classes, threshold=0.5,
                                               order=Order.OVERLAP_OUTPUT_TO_TARGET)

        y_score, y_true = np.array(y_score), np.array(y_true)

        # from binary to multi class
        lb = preprocessing.LabelBinarizer()
        lb.fit(classes)
        y_score = lb.inverse_transform(y_score)
        y_true = lb.inverse_transform(y_true)

        y_score_true = np.stack((np.array(y_score), np.array(y_true)), axis=-1)

        assert len(np.where((y_score_true == (1, 1)).all(axis=1))[0]) == 1  # TP
        assert len(np.where((y_score_true == (1, 0)).all(axis=1))[0]) == 1  # FP
        assert len(np.where((y_score_true == (0, 1)).all(axis=1))[0]) == 1  # FN

    def test_find_correspondences_overlap_binary_4(self):
        classes = [0, 1]
        y_score, y_true = find_correspondences(outputs=[self.box13, self.box14],
                                               targets=[self.box11, self.box12], classes=classes, threshold=0.5,
                                               order=Order.OVERLAP_TARGET_TO_OUTPUT)

        y_score, y_true = np.array(y_score), np.array(y_true)

        # from binary to multi class
        lb = preprocessing.LabelBinarizer()
        lb.fit(classes)
        y_score = lb.inverse_transform(y_score)
        y_true = lb.inverse_transform(y_true)

        y_score_true = np.stack((np.array(y_score), np.array(y_true)), axis=-1)

        assert len(np.where((y_score_true == (1, 1)).all(axis=1))[0]) == 1  # TP
        assert len(np.where((y_score_true == (1, 0)).all(axis=1))[0]) == 1  # FP
        assert len(np.where((y_score_true == (0, 1)).all(axis=1))[0]) == 1  # FN

    def test_find_correspondences_overlap_binary_5(self):
        classes = [0, 1]
        y_score, y_true = find_correspondences(outputs=[self.box14, self.box13],
                                               targets=[self.box12, self.box11], classes=classes, threshold=0.5,
                                               order=Order.OVERLAP_TARGET_TO_OUTPUT)

        y_score, y_true = np.array(y_score), np.array(y_true)

        # from binary to multi class
        lb = preprocessing.LabelBinarizer()
        lb.fit(classes)
        y_score = lb.inverse_transform(y_score)
        y_true = lb.inverse_transform(y_true)

        y_score_true = np.stack((np.array(y_score), np.array(y_true)), axis=-1)

        assert len(np.where((y_score_true == (1, 1)).all(axis=1))[0]) == 2  # TP
        assert len(np.where((y_score_true == (1, 0)).all(axis=1))[0]) == 0  # FP
        assert len(np.where((y_score_true == (0, 1)).all(axis=1))[0]) == 0  # FN


    def test_intersection_over_union(self):
        # prepare
        iou = IoU()
        box1 = [dict(c=np.array([0., 1., 4.]),
                     d=np.array([4., 8., 10.]),
                     r=Rotation.from_euler('xyz', [0, 0, 30], degrees=True).as_dcm()),
                dict(c=np.array([0., 1., 4.]),
                     d=np.array([4., 8., 10.]),
                     r=Rotation.from_euler('xyz', [0, 0, 30], degrees=True).as_dcm()),
                dict(c=np.array([0., 1., 4.]),
                     d=np.array([4., 8., 10.]),
                     r=Rotation.from_euler('xyz', [0, 0, 30], degrees=True).as_dcm())]
        box2 = [dict(c=np.array([0., 1., 2]),
                     d=np.array([4., 8., 10.]),
                     r=Rotation.from_euler('xyz', [0, 0, 30], degrees=True).as_dcm()),
                dict(c=np.array([0., 1., 4.]),
                     d=np.array([4., 8., 10.]),
                     r=Rotation.from_euler('xyz', [0, 0, 30], degrees=True).as_dcm()),
                dict(c=np.array([20., 1., 4.]),
                     d=np.array([4., 8., 10.]),
                     r=Rotation.from_euler('xyz', [45, 45, 30], degrees=True).as_dcm())]
        # test
        result = iou(box1, box2)
        expected_res = [4 / 6, 1, 0]
        assert np.isclose(result, expected_res).all()

