import numpy as np
from numpy import sin, cos, sqrt

def transformation_matrix(offset, scsv, sc, psi):
    """Produces a transformation matrix for the WIC camera using axes' orientations.
    Taken from fuv_rotation_matrix.pro from fuview.
    """
    t1a = np.array([
        [cos(offset[2]), -sin(offset[2]), 0],
        [sin(offset[2]), cos(offset[2]), 0],
        [0, 0, 1]
    ])
    t1b = np.array([
        [1, 0, 0],
        [0, -sin(offset[1]), -cos(offset[1])],
        [0, cos(offset[1]), -sin(offset[1])]
    ])
    t1c = np.array([
        [sin(offset[0]), cos(offset[0]), 0],
        [-cos(offset[0]), sin(offset[0]), 0],
        [0, 0, 1]
    ])
    t1 = np.matmul(t1c, np.matmul(t1b, t1a))

    # Seems to be resued multiple times
    sqrt_term = sqrt(1 - scsv[0] ** 2)

    sin_alp = scsv[1] / sqrt_term
    cos_alp = scsv[2] / sqrt_term
    sin_bet = scsv[0]
    cos_bet = sqrt_term

    t2a = np.array([
        [1, 0, 0],
        [0, cos_alp, -sin_alp],
        [0, sin_alp, cos_alp]
    ])
    t2b = np.array([
        [cos_bet, 0, -sin_bet],
        [0, 1, 0],
        [sin_bet, 0, cos_bet]
    ])
    t2 = np.matmul(t2b, t2a)

    cos_eta = sc[2] / sqrt_term
    sin_eta = sqrt(1 - scsv[0] ** 2 - sc[2] ** 2) / sqrt_term
    cos_del = (scsv[0] * sc[0] + sc[1] * sqrt(1 - scsv[0] ** 2 - sc[2] ** 2)) / (1 - sc[2] ** 2)
    sin_del = (scsv[0] * sc[1] - sc[0] * sqrt(1 - scsv[0] ** 2 - sc[2] ** 2)) / (1 - sc[2] ** 2)

    t3a = np.array([
        [cos(psi), -sin(psi), 0],
        [sin(psi), cos(psi), 0],
        [0, 0, 1]
    ])
    t3b = np.array([
        [cos_bet, 0, sin_bet],
        [0, 1, 0],
        [-sin_bet, 0, cos_bet]
    ])
    t3c = np.array([
        [1, 0, 0],
        [0, cos_eta, sin_eta],
        [0, -sin_eta, cos_eta]
    ])
    t3d = np.array([
        [cos_del, -sin_del, 0],
        [sin_del, cos_del, 0],
        [0, 0, 1]
    ])

    t3 = np.matmul(t3d, np.matmul(t3c, np.matmul(t3b, t3a)))

    return np.matmul(t3, np.matmul(t2, t1))
