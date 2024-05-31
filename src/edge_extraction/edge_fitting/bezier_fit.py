import numpy as np
from scipy.optimize import curve_fit


def bezier_curve(tt, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11):
    n = len(tt)
    matrix_t = np.concatenate(
        [(tt**3)[..., None], (tt**2)[..., None], tt[..., None], np.ones((n, 1))],
        axis=1,
    ).astype(float)
    matrix_w = np.array(
        [[-1, 3, -3, 1], [3, -6, 3, 0], [-3, 3, 0, 0], [1, 0, 0, 0]]
    ).astype(float)
    matrix_p = np.array(
        [[p0, p1, p2], [p3, p4, p5], [p6, p7, p8], [p9, p10, p11]]
    ).astype(float)
    return np.dot(np.dot(matrix_t, matrix_w), matrix_p).reshape(-1)


def bezier_fit(xyz, error_threshold=1.0):
    n = len(xyz)
    t = np.linspace(0, 1, n)
    xyz = xyz.reshape(-1)

    popt, _ = curve_fit(bezier_curve, t, xyz)

    # Generate fitted curve
    fitted_curve = bezier_curve(t, *popt).reshape(-1, 3)

    # Calculate residuals
    residuals = xyz.reshape(-1, 3) - fitted_curve

    # Calculate RMSE
    rmse = np.sqrt(np.mean(np.sum(residuals**2, axis=1)))

    if rmse > error_threshold:
        return None
    else:
        return popt
