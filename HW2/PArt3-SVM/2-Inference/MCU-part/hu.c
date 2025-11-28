#include "hu.h"
#include <math.h>
#include <string.h> // For memset
#include <float.h>

/**
 * @brief Calculates Hu moment invariants from a single-channel image.
 *        Works for MNIST-style grayscale images. Numerically stable.
 */
int calculate_hu_moments(
    const float* image_pixels,
    int image_rows,
    int image_cols,
    float* hu_moments_out)
{
    if (!image_pixels || !hu_moments_out || image_rows <= 0 || image_cols <= 0) {
        return -2; // Invalid arguments
    }

    // Accumulate moments in double for precision
    double m[4][4];
    double mu[4][4];
    double nu[4][4];
    memset(m, 0, sizeof(m));
    memset(mu, 0, sizeof(mu));
    memset(nu, 0, sizeof(nu));

    // 1. Compute raw moments m[p][q]
    for (int r = 0; r < image_rows; r++) {
        for (int c = 0; c < image_cols; c++) {
            double I = image_pixels[r*image_cols + c];
            if (I > 0.0) {
                double c_pow[4] = {1.0, (double)c, (double)c*c, (double)c*c*c};
                double r_pow[4] = {1.0, (double)r, (double)r*r, (double)r*r*r};
                for (int i = 0; i < 4; i++)
                    for (int j = 0; j < 4; j++)
                        m[i][j] += I * c_pow[i] * r_pow[j];
            }
        }
    }

    if (m[0][0] == 0.0) { // Empty image
        for (int i = 0; i < HU_MOMENTS_COUNT; i++)
            hu_moments_out[i] = 0.0f;
        return -1;
    }

    // 2. Centroid
    double cx = m[1][0] / m[0][0];
    double cy = m[0][1] / m[0][0];

    // 3. Central moments (pixel-based for accuracy)
    mu[0][0] = m[0][0];
    mu[1][0] = mu[0][1] = 0.0;
    mu[2][0] = mu[0][2] = mu[1][1] = 0.0;
    mu[3][0] = mu[0][3] = mu[2][1] = mu[1][2] = 0.0;

    for (int r = 0; r < image_rows; r++) {
        for (int c = 0; c < image_cols; c++) {
            double I = image_pixels[r*image_cols + c];
            double xr = c - cx;
            double yr = r - cy;

            mu[2][0] += xr*xr * I;
            mu[0][2] += yr*yr * I;
            mu[1][1] += xr*yr * I;
            mu[3][0] += xr*xr*xr * I;
            mu[0][3] += yr*yr*yr * I;
            mu[2][1] += xr*xr*yr * I;
            mu[1][2] += xr*yr*yr * I;
        }
    }

    // 4. Scale-invariant moments
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            if (i + j >= 2)
                nu[i][j] = mu[i][j] / pow(mu[0][0], (i + j)/2.0 + 1.0);

    // 5. Hu invariants
    hu_moments_out[0] = (float)(nu[2][0] + nu[0][2]);
    hu_moments_out[1] = (float)(pow(nu[2][0] - nu[0][2],2) + 4*pow(nu[1][1],2));
    hu_moments_out[2] = (float)(pow(nu[3][0] - 3*nu[1][2],2) + pow(3*nu[2][1] - nu[0][3],2));
    hu_moments_out[3] = (float)(pow(nu[3][0] + nu[1][2],2) + pow(nu[2][1] + nu[0][3],2));
    hu_moments_out[4] = (float)((nu[3][0]-3*nu[1][2])*(nu[3][0]+nu[1][2])*(pow(nu[3][0]+nu[1][2],2)-3*pow(nu[2][1]+nu[0][3],2)) +
                                (3*nu[2][1]-nu[0][3])*(nu[2][1]+nu[0][3])*(3*pow(nu[3][0]+nu[1][2],2)-pow(nu[2][1]+nu[0][3],2)));
    hu_moments_out[5] = (float)((nu[2][0]-nu[0][2])*(pow(nu[3][0]+nu[1][2],2)-pow(nu[2][1]+nu[0][3],2)) +
                                4*nu[1][1]*(nu[3][0]+nu[1][2])*(nu[2][1]+nu[0][3]));
    hu_moments_out[6] = (float)((3*nu[2][1]-nu[0][3])*(nu[3][0]+nu[1][2])*(pow(nu[3][0]+nu[1][2],2)-3*pow(nu[2][1]+nu[0][3],2)) -
                                (nu[3][0]-3*nu[1][2])*(nu[2][1]+nu[0][3])*(3*pow(nu[3][0]+nu[1][2],2)-pow(nu[2][1]+nu[0][3],2)));

    return 0;
}
