import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """
    T = np.asarray(T)
    points = np.asarray(points)
    if len(points.shape) == 1:
        points = points.reshape(1, -1)
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    print(points)
    points = (T @ points.T).T[:, :-1]
    return points.squeeze()
    pass