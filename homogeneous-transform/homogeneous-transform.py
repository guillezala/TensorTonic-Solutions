import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """
    points = np.array(points)
    single = points.ndim == 1

    if single:
        points = points.reshape(1, 3)

    ones = np.ones((points.shape[0], 1))
    points_h = np.concatenate((points, ones), axis=1)

    print("shape points", points_h.shape)
    print("shape T", len(T))

    result = (T @ points_h.T).T[:, :-1]

    return result.reshape(3,) if single else result