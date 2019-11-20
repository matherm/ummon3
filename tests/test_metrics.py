# -*- coding: utf-8 -*-
# @Author: Daniel Dold
# @Date:   2019-11-20 15:42:51
# @Last Modified by:   Daniel Dold
# @Last Modified time: 2019-11-20 16:12:35
import numpy as np
from scipy.spatial.transform import Rotation
from ummon.metrics import *
from ummon.metrics.geometric_metrics import halfspace_representation, intersection
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
