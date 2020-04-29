# -*- coding: utf-8 -*-
# @Author: daniel
# @Date:   2020-04-28 17:38:39
# @Last Modified by:   daniel
# @Last Modified time: 2020-04-29 10:02:37
from scipy.spatial.transform import Rotation
import numpy as np

__all__ = ['Transform', 'transform_bbox', 'transformation_from_plane']


class Transform():
    """
    This class describes a transformation. It could be used to transfrom 3D
    points into another coordinate system.
    """

    def __init__(self, rotation_a_b=[0, 0, 0, 1.], translation_a_b=[0, 0, 0]):
        """Summary

        Parameters
        ----------
        rotation_a_b : list, optional
            describes the rotation from a to b. Defined in coo_a
        translation_a_b : list, optional
            describes the translation from a to b (centre of coo_b from coo_a perspective)
        """
        super().__init__()
        self.quaternion_ = rotation_a_b
        rot_a_b = Rotation.from_quat(rotation_a_b)
        t = np.zeros((4, 4), dtype=np.float64)
        t[0:3, 0:3] = rot_a_b.as_dcm()  # for scipy > 1.4 as_matrix()
        t[0:3, 3] = translation_a_b
        t[3, 3] = 1.0
        self.t_ = t.transpose()

    def __call__(self, p) -> np.array:
        """Apply the transformation. 
        It's possible to transforme a singel vector shape (4,) or multiple vectors shape (n,4) at once.

        Parameters
        ----------
        p : nx4
            vectors described in coordinate system b

        Returns
        -------
        np.array [nx4]
            returns the transformed vector in coordinate system a
        """
        res = np.matmul(p, self.t_)
        if len(res.shape) == 1:
            return res / res[-1]
        else:
            return res / res[:, -1:]

    def inv(self):
        """returns the inverse transformation 'T_b_a'

        Returns
        -------
        Transform
            the inverse transforamtion instance
        """
        r_b_a = self.get_rotation().inv()
        t_b_a = -r_b_a.apply(self.get_translation())
        return Transform(r_b_a.as_quat(), t_b_a)

    def get_rotation(self) -> Rotation:
        """get the rotation part

        Returns
        -------
        Rotation
            the rotation 'r_a_b'
        """
        return Rotation.from_quat(self.quaternion_)

    def get_translation(self) -> np.array:
        """get the translation part

        Returns
        -------
        np.array
            the translation 't_a_b'
        """
        return self.t_[3, 0:3]

    def __repr__(self):
        return "Rotation(quat): {}, Translation: {}".format(self.get_rotation().as_quat(), self.get_translation())


def transform_bbox(bbox: dict, trans: Transform=Transform([0, 0, 0, 1.], [0, 0, 0.])) -> dict:
    """transform a boundingbox according the coordinate transformation 

    Parameters
    ----------
    bbox : dict
        bbox in coordinate_b
        {   c: centre (3,)
                        d: dimension (3,)
                        r: rotation in quaternion (4,)
        }
    trans : Transform, optional
        The coordinate transformation t_a_b 

    Returns
    -------
    dict
        bbox in coordinate_a
    """
    bbox_trans = dict(d=np.array(bbox['d']))
    bbox_trans['c'] = trans(np.append(bbox['c'], [1.]))[0:3]
    bbox_trans['r'] = (trans.get_rotation() *
                       Rotation.from_quat(bbox['r'])).as_quat()
    return bbox_trans


def transformation_from_plane(plane_coo: list, axis=2) -> Transform:
    """computes the transformation (R|t) according the plane parameters.
    This defines a new coordinate system where the plane is defined by two coordinate axis.

    Parameters
    ----------
    plane_coo : list or np.array (2,3)
        [n, r0] defines the plane in point-normal form n*(r-r0) = 0
    axis : int, optional
        select the axis that defines the plane normal in the new coordinate system. (The other two defines the plane)

    Returns
    -------
    Transform
        return the transformation T_p_coo, defined from the plane to the coordinate system in which the plane parameters were defined
    """
    normal = plane_coo[0]
    # check if normal is facing upwards.
    # check normals facing. n*(vp -p) > 0 (viewpoint is center)
    if np.dot(normal, -np.array([1., 0., 0.])) < 0:  # flip normal
        normal *= -1.
    r0 = plane_coo[1]
    aligned_vec = np.zeros(3)
    translation_p_coo = np.zeros(3)
    aligned_vec[axis] = 1.
    assert np.isclose(np.linalg.norm(normal),
                      1.0), "Length of normal must be 1.0"

    # return no transformation if normal and aligned axis are to close
    if np.isclose(np.dot(normal, aligned_vec), 0.):
        print("attention transformation could not be computed")
        return Transform([0., 0., 0., 1.], translation_p_coo)

    translation_p_coo[axis] = -np.dot(normal, r0)

    rot_vec = np.cross(normal, aligned_vec)
    rot_vec /= np.linalg.norm(rot_vec)
    rot_vec *= np.arccos(np.dot(normal, aligned_vec))

    rotation_p_coo = Rotation.from_rotvec(rot_vec).as_quat()
    return Transform(rotation_p_coo, translation_p_coo)


def usage_example():
    # Transformation
    # test transformation
    r_a_b = Rotation.from_euler('xyz', [90, 0, 0], degrees=True)
    t_a_b = np.array([-10, 0, 0])
    transform_a_b = Transform(r_a_b.as_quat(), t_a_b)
    # test vectors
    vec1 = np.array([1, 2, 3, 1])
    vec2 = np.arange(12).reshape((-1, 4))
    vec2[:, -1] = 1

    print("vec_b: {}, vec_a: {}".format(vec1, transform_a_b(vec1)))
    print("vec_b: \n{}\n, vec_a: \n{}".format(vec2, transform_a_b(vec2)))
    # show invers function
    transform_b_a = transform_a_b.inv()
    print("vec_a: {}, vec_b: {}".format(transform_a_b(
        vec1), transform_b_a(transform_a_b(vec1))))
    print("vec_a: \n{}\n, vec_b: \n{}".format(
        transform_a_b(vec2), transform_b_a(transform_a_b(vec2))))

    # plane from transformation
    plane = np.array([[0, np.sqrt(0.5), np.sqrt(0.5)], [0, 0, 10]])
    t = transformation_from_plane(plane)
    print("rotation from plane: ", t.get_rotation().as_euler('xyz', True))


if __name__ == '__main__':
    usage_example()
