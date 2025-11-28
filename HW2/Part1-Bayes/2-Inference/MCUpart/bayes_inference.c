/*
 * bayes_inference.c
 *
 *  Created on: Oct 6, 2023
 *  Authors: Berkan Höke, Eren Atmaca
 *  Updated: Nov 2025 by ChatGPT (CMSIS-DSP / STM32F7 compatible)
 */

#include "bayes_inference.h"
#include <math.h>
#include <string.h>

int8_t BAYES_Classify(arm_matrix_instance_f32 *input, int case_type, arm_matrix_instance_f32 *output)
{
    int8_t status = ARM_MATH_SUCCESS;
    float discr[NUM_CLASSES] = {0};

    /* Transpose input vector (x^T) */
    float32_t input_T_buf[NUM_FEATURES];
    arm_matrix_instance_f32 input_T = {1, NUM_FEATURES, input_T_buf};
    status = arm_mat_trans_f32(input, &input_T);

    for (int cls = 0; cls < NUM_CLASSES; cls++)
    {
        arm_matrix_instance_f32 mu = {NUM_FEATURES, 1, &MEANS[cls][0]};
        arm_matrix_instance_f32 sigma_inv = {NUM_FEATURES, NUM_FEATURES, &INV_COVS[cls][0][0]};

        /* Temporary local buffers */
        float32_t xt_sigma_buf[NUM_FEATURES];
        float32_t sigma_mu_buf[NUM_FEATURES];

        float32_t xt_sigma_x = 0.0f;
        float32_t sigma_mu_x = 0.0f;
        float32_t mu_sigma_mu = 0.0f;

        /* Define matrix instances */
        arm_matrix_instance_f32 xt_sigma = {1, NUM_FEATURES, xt_sigma_buf};
        arm_matrix_instance_f32 sigma_mu = {NUM_FEATURES, 1, sigma_mu_buf};
        arm_matrix_instance_f32 xt_sigma_x_mat = {1, 1, &xt_sigma_x};
        arm_matrix_instance_f32 sigma_mu_x_mat = {1, 1, &sigma_mu_x};
        arm_matrix_instance_f32 mu_sigma_mu_mat = {1, 1, &mu_sigma_mu};

        float prior_log = logf(CLASS_PRIORS[cls]);
        float log_det = logf(DETS[cls]) * (-0.5f);

        /* ---- Case 3: Full Quadratic Discriminant ---- */
        if (case_type == 3)
        {
            // (x^T * Σ⁻¹ * x)
            status += arm_mat_mult_f32(&input_T, &sigma_inv, &xt_sigma);
            status += arm_mat_mult_f32(&xt_sigma, input, &xt_sigma_x_mat);
            xt_sigma_x *= -0.5f;

            // (μ^T * Σ⁻¹ * x)
            status += arm_mat_mult_f32(&sigma_inv, &mu, &sigma_mu);
            arm_matrix_instance_f32 sigma_mu_T = {1, NUM_FEATURES, sigma_mu_buf};
            status += arm_mat_mult_f32(&sigma_mu_T, input, &sigma_mu_x_mat);

            // (μ^T * Σ⁻¹ * μ)
            arm_matrix_instance_f32 mu_T = {1, NUM_FEATURES, &MEANS[cls][0]};
            status += arm_mat_mult_f32(&mu_T, &sigma_mu, &mu_sigma_mu_mat);

            discr[cls] = xt_sigma_x + sigma_mu_x - 0.5f * mu_sigma_mu + log_det + prior_log;
        }
        else
        {
            return ARM_MATH_ARGUMENT_ERROR; // unsupported case
        }
    }

    memcpy(output->pData, discr, sizeof(float32_t) * NUM_CLASSES);
    output->numRows = 1;
    output->numCols = NUM_CLASSES;

    return status;
}


/**
 * @brief Normalize discriminant scores using softmax
 */
void BAYES_Softmax(arm_matrix_instance_f32 *output)
{
    float32_t max_val = output->pData[0];
    for (int i = 1; i < NUM_CLASSES; i++)
        if (output->pData[i] > max_val) max_val = output->pData[i];

    float32_t sum = 0.0f;
    for (int i = 0; i < NUM_CLASSES; i++)
    {
        output->pData[i] = expf(output->pData[i] - max_val);
        sum += output->pData[i];
    }

    for (int i = 0; i < NUM_CLASSES; i++)
        output->pData[i] /= sum;
}
