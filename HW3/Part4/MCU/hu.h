#ifndef HU_H
#define HU_H

/**
 * @file hu.h
 * @brief Public interface for calculating Hu Moment Invariants from an image.
 *
 * This module provides a function to compute a set of seven shape descriptors
 * that are invariant to translation, scale, and rotation.
 *
 * DEPENDENCIES:
 * Requires linking with the standard math library (e.g., using -lm with gcc).
 */

#define HU_MOMENTS_COUNT 7 // The number of Hu moment invariants

/**
 * @brief Calculates the seven Hu moment invariants for a single-channel image.
 *
 * This function processes a flat array of pixel data to produce a feature vector
 * that describes the shape of the objects within it.
 *
 * @param image_pixels Pointer to a flat array of image pixel data (single channel, float).
 * @param image_rows The number of rows in the image.
 * @param image_cols The number of columns in the image.
 * @param hu_moments_out A pointer to a float array of at least size HU_MOMENTS_COUNT,
 *                       which will be populated with the calculated Hu moments.
 * @return 0 on success.
 * @return -1 if the image is empty (total intensity is zero), preventing division-by-zero.
 * @return -2 for invalid arguments (null pointers, non-positive dimensions).
 */
int calculate_hu_moments(
    const float* image_pixels,
    int image_rows,
    int image_cols,
    float* hu_moments_out
);

#endif // HU_H