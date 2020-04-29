# -*- coding: utf-8 -*-
# @Author: daniel
# @Date:   2020-04-29 07:35:52
# @Last Modified by:   daniel
# @Last Modified time: 2020-04-29 10:08:22
import pytest
from ummon.ip.transform_utils import *
from scipy.spatial.transform import Rotation
import numpy as np


class TestTransformUtils():
    """docstring for TestTransformUtils"""

    def test_transformation_single(self):
        # test single vec translation
        r_a_b = Rotation.from_euler('xyz', [0, 0, 0], degrees=True)
        t_a_b = np.array([1, 2, 3])
        transform_a_b = Transform(r_a_b.as_quat(), t_a_b)
        vec1_b = np.array([0, 0, 0, 1])
        assert np.isclose(transform_a_b(vec1_b), [1, 2, 3, 1]).all()

    def test_transformation_multiple(self):
        # test rotation multiple vecs
        r_a_b = Rotation.from_euler('xyz', [90, 0, 0], degrees=True)
        t_a_b = np.array([1, 2, 3])
        transform_a_b = Transform(r_a_b.as_quat(), t_a_b)
        vec2_b = np.arange(8).reshape((-1, 4))
        vec2_b[:, -1] = 1
        vec2_a_exp = np.array([[1, 0, 4, 1],
                               [5, -4, 8, 1]])
        assert np.allclose(transform_a_b(vec2_b), vec2_a_exp)

    def test_transformation_inverse(self):
        # test inv
        r_a_b = Rotation.from_euler('xyz', [90, 45, 30], degrees=True)
        t_a_b = np.array([1, -2, 3])
        transform_a_b = Transform(r_a_b.as_quat(), t_a_b)
        vec2_b = np.arange(16).reshape((-1, 4))
        vec2_b[:, -1] = 1
        assert np.allclose(transform_a_b.inv()(transform_a_b(vec2_b)), vec2_b)

    def test_transformation_get_functions(self):
        r_a_b = Rotation.from_euler('xyz', [22, 45, 30], degrees=True)
        t_a_b = np.array([1, -2, 3])
        assert np.allclose(
            Transform(r_a_b.as_quat(), t_a_b).get_rotation().as_quat(), r_a_b.as_quat())
        assert np.allclose(
            Transform(r_a_b.as_quat(), t_a_b).get_translation(), t_a_b)


class TestTransfromFromPlane():
    def test_transform(self):
        plane_coo = np.array([[0, np.sqrt(0.5), np.sqrt(0.5)],
                              [10, 0, 10]])
        T_p_coo = transformation_from_plane(plane_coo)
        assert np.allclose(T_p_coo.get_rotation(
        ).as_euler('xyz', True), [45, 0, 0])
        assert np.allclose(T_p_coo.get_translation(), [0, 0, -10/np.sqrt(2)])
