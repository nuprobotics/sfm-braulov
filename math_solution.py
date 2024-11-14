import numpy as np


def create_projection_matrix(K, R, t):
    R_T = R.T

    t_transformed = -R_T @ t

    P = K @ np.hstack((R_T, t_transformed.reshape(-1, 1)))
    return P


def triangulation(camera_matrix, camera_position1, camera_rotation1, camera_position2, camera_rotation2, image_points1,
                  image_points2):

    P1 = create_projection_matrix(camera_matrix, camera_rotation1, camera_position1)
    P2 = create_projection_matrix(camera_matrix, camera_rotation2, camera_position2)


    num_points = image_points1.shape[0]
    points_3d = np.zeros((num_points, 3))


    for i in range(num_points):
        x1, y1 = image_points1[i]
        x2, y2 = image_points2[i]


        A = np.array([
            (x1 * P1[2] - P1[0]),
            (y1 * P1[2] - P1[1]),
            (x2 * P2[2] - P2[0]),
            (y2 * P2[2] - P2[1])
        ])


        _, _, Vt = np.linalg.svd(A)
        X_homogeneous = Vt[-1]


        points_3d[i] = X_homogeneous[:3] / X_homogeneous[3]

    return points_3d
