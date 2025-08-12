import numpy as np
from scipy.ndimage import convolve

def fast_morans_i_with_nans(grid):
    """
    Computes Moran's I for 2D grid data with NaN values using convolution.
    
    Parameters:
        grid (np.ndarray): 2D array of values (may contain NaN)
    
    Returns:
        Moran's I (float), or NaN if insufficient non-NaN data
    """
    mask = ~np.isnan(grid)
    valid_count = np.sum(mask)
    if valid_count < 2:
        return np.nan    
    grid_filled = np.where(mask, grid, 0)
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=float)
    neighbor_count = convolve(mask.astype(float), kernel, 
                            mode='constant', cval=0)
    spatial_lag_sum = convolve(grid_filled, kernel, 
                             mode='constant', cval=0)
    spatial_lag = np.divide(spatial_lag_sum, neighbor_count,
                           out=np.zeros_like(grid),
                           where=(neighbor_count > 0))
    x_mean = grid[mask].mean()
    x_dev = np.where(mask, grid - x_mean, 0)
    numerator = np.sum(x_dev * (spatial_lag - x_mean))
    denominator = np.sum(x_dev[mask] ** 2)
    if denominator == 0 or np.sum(neighbor_count[mask]) == 0:
        return np.nan    
    I = (valid_count / np.sum(neighbor_count[mask])) * (numerator / denominator)
    return I