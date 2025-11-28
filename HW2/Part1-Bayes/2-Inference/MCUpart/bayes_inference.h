/*
 * bayes_inference.h
 *
 *  Created on: Oct 6, 2023
 *  Authors: Berkan Höke, Eren Atmaca
 *  Updated: Nov 2025 by ChatGPT (STM32F7/CMSIS-DSP Compatible)
 */

#ifndef INC_BAYES_INFERENCE_H_
#define INC_BAYES_INFERENCE_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "arm_math.h"
#include "bayes_cls_config.h"

/**
 * @brief  Compute Bayesian class discriminants (log-probabilities)
 * @param  input:  pointer to input feature vector (NUM_FEATURES x 1)
 * @param  case_type: discriminant type (1=linear, 2=diag, 3=full quadratic)
 * @param  output: pointer to output vector (1 x NUM_CLASSES)
 * @retval ARM_MATH_SUCCESS on success, ARM_MATH_ARGUMENT_ERROR otherwise
 */
int8_t BAYES_Classify(arm_matrix_instance_f32 *input, int case_type, arm_matrix_instance_f32 *output);

/**
 * @brief  Optional: normalize discriminant outputs to softmax probabilities
 * @param  output:  input/output matrix (1 x NUM_CLASSES)
 */
void BAYES_Softmax(arm_matrix_instance_f32 *output);

#ifdef __cplusplus
}
#endif

#endif /* INC_BAYES_INFERENCE_H_ */
