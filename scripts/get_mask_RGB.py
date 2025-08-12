import sys
import os
import numpy as np
import pandas as pd
import pickle
import argparse
import matplotlib.pyplot as plt
from utils import save_image, save_pickle, load_pickle, load_mask, load_image
from extract_features import patchify
from PIL import Image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix', type=str)
    parser.add_argument('--m', type=float, default=1)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    prefix = args.prefix
    m = args.m

    he = load_image(prefix + 'he.jpg')
    he_tiles, shapes = patchify(he, patch_size=16)
    he_mean = [np.mean(he_tile) for he_tile in he_tiles]
    he_mean_image = np.reshape(he_mean, shapes['tiles'])
    he_std = [np.std(he_tile) for he_tile in he_tiles]
    he_std_image = np.reshape(he_std, shapes['tiles'])

    # titlesize = 1
    # plt.figure(figsize=(16, 10))
    # plt.subplot(1, 2, 1)
    # filtered_he_mean_image = he_mean_image.copy()
    # # filtered_he_mean_image[~conserve_index_image] = np.nan
    # plt.imshow(filtered_he_mean_image)
    # _ = plt.title('Averaged RGB Image', fontsize=titlesize)
    # plt.subplot(1, 2, 2)
    # filtered_he_std_image = he_std_image.copy()
    # # filtered_he_std_image[~conserve_index_image] = np.nan
    # plt.imshow(filtered_he_std_image)
    # _ = plt.title('RGB Std Image', fontsize=titlesize)

    mean_intensity = he_mean_image.copy().flatten()
    std_dev = he_std_image.copy().flatten()

    # Step 1: Fit a quadratic model using np.polyfit (mean_intensity, std_dev)
    coeffs = np.polyfit(mean_intensity, std_dev, 2)  # Fit a quadratic polynomial
    a, b, c = coeffs  # Coefficients of the quadratic model
    # Step 2: Calculate the x-coordinate of the vertex (peak) of the parabola
    x_vertex = -b / (2 * a)
    # Step 3: Calculate the y-coordinate (standard deviation) at the vertex
    y_vertex = a * x_vertex ** 2 + b * x_vertex + c

    # for visualization
    def linear_line(x, x_vertex, m):
        return m * x - x_vertex
    print(f"Peak of the parabola occurs at Mean Intensity: {x_vertex:.2f}, Standard Deviation: {y_vertex:.2f}")

    def linear_bound(mean_RGB, std_RGB, m):
        coeffs = np.polyfit(mean_RGB, std_RGB, 2)  # Fit quadratic model based on input data
        a, b, c = coeffs
        # Calculate the x-coordinate of the vertex (peak)
        x_vertex = -b / (2 * a)
        # Calculate the y-intercept for the line with slope m that passes through (x_vertex, 0)
        y_intercept = -m * x_vertex
        print(f"Peak of the parabola occurs at Mean Intensity: {x_vertex:.2f}")
        return m * mean_RGB - x_vertex

    # # Calculate the y values of the linear boundary for each mean_intensity
    # linear_boundary = linear_bound(mean_intensity, std_dev, m)
    # # Step 1: Create a mask to identify points below the linear boundary
    # below_boundary_mask = std_dev < linear_boundary
    # # Step 2: Save the indices of the points below the linear boundary
    # below_boundary_indices = np.where(below_boundary_mask)[0]
    # Step 3: Plot the points, color those below the boundary in red
    # plt.scatter(mean_intensity[~below_boundary_mask], std_dev[~below_boundary_mask], color='blue', s=.1)
    # plt.scatter(mean_intensity[below_boundary_mask], std_dev[below_boundary_mask], color='red', s=.1)
    # # Plot the quadratic curve for visualization
    # x_vals = np.linspace(np.min(mean_intensity), np.max(mean_intensity), 500)
    # y_vals = a * x_vals ** 2 + b * x_vals + c
    # plt.plot(x_vals, y_vals, 'g--', label='Fitted Quadratic')
    # # Mark the vertex (peak)
    # plt.scatter([x_vertex], [y_vertex], color='green', s=50, zorder=5)
    #
    # # Plot the linear boundary line
    # x_line = np.linspace(x_vertex, np.max(mean_intensity), 500)
    # y_line = linear_line(x_line, x_vertex, m)
    # plt.plot(x_line, y_line, 'r--', label='Linear Boundary')
    # plt.xlim(0, he_mean_image.max() + 10)
    # plt.ylim(-5, he_std_image.max() + 25)
    # # Plot settings
    # plt.xlabel('Mean Intensity')
    # plt.ylabel('Standard Deviation')
    # plt.title(f'Linear Boundary with M = {m}')
    # plt.legend()
    # plt.show()

    linear_boundary = linear_bound(mean_intensity, std_dev, m)
    # Superpixels where he_std is below the boundary will be marked as to remove
    conserve_index = [True if he_std[i] >= linear_boundary[i] else False for i in range(len(he_tiles))]
    # save_pickle(conserve_index, f'filterRGB/conserve_index_linearBoundary_m{m}.pickle')
    conserve_index_image = np.reshape(conserve_index, shapes['tiles'])

    # Display results
    # plt.figure(figsize=(60, 50))
    # plt.subplot(1, 2, 1)
    # filtered_he_mean_image = he_mean_image.copy()
    # filtered_he_mean_image[~conserve_index_image] = np.nan
    # plt.imshow(filtered_he_mean_image)
    # _ = plt.title('Filtered Averaged RGB Image', fontsize=titlesize)
    # plt.subplot(1, 2, 2)
    # filtered_he_std_image = he_std_image.copy()
    # filtered_he_std_image[~conserve_index_image] = np.nan
    # plt.imshow(filtered_he_std_image)
    # _ = plt.title('Filtered RGB std Image', fontsize=titlesize)
    # plt.savefig(f'filterRGB/Filtered_LinearBoundary_m{m}_RGB_image.png', dpi=500, bbox_inches='tight')
    # plt.show()

    image_height, image_width = he.shape[0], he.shape[1]
    super_pixel_size = 16
    # Create a numpy array to hold the mask, initially all zeros
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    # Reshape the True/False list to match the super-pixel grid layout
    num_super_pixels_y = image_height // super_pixel_size
    num_super_pixels_x = image_width // super_pixel_size
    # Convert the True/False list into a 2D array corresponding to the super-pixel grid
    super_pixel_values = np.array(conserve_index).reshape((num_super_pixels_y, num_super_pixels_x))
    # Populate the mask based on the values of the super-pixel grid
    for i in range(num_super_pixels_y):
        for j in range(num_super_pixels_x):
            value = 255 if super_pixel_values[i, j] else 0  # Use 255 for white, 0 for black
            # Set the corresponding region in the mask
            mask[i * super_pixel_size:(i + 1) * super_pixel_size,
            j * super_pixel_size:(j + 1) * super_pixel_size] = value
    mask_image = Image.fromarray(mask)
    save_image(super_pixel_values, prefix+"mask-small-RGB.png")
    mask_image.save(prefix+"mask-full-RGB.png")

if __name__ == '__main__':
    main()



