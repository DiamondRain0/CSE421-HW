#ifndef SVM_LOGIC_H
#define SVM_LOGIC_H

#include "svm_digits_config.h"

// Predicts digit (0-9) from 7 Hu Moment features
int svm_predict_digit(float *raw_features);

#endif