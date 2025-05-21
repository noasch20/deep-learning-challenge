# Alphabet Soup Deep Learning Model Report

## Overview
The purpose of this analysis was to develop and optimize a deep learning neural network using TensorFlow and Keras to predict whether applicants to Alphabet Soup, a fictional nonprofit, will be successful recipients of funding.
By preprocessing the dataset and experimenting with different network architectures and configurations, the goal was to achieve a classification model with at least 75% accuracy.

## Results
### Data Preprocessing
- Target Variable
  - `IS_SUCCESSFUL` -- A binary indicator of whether the applicant received funding.
- Feature Variables
  - One-hot encoded categorical features such as `APPLICATION_TYPE`, `AFFILIATION`, `INCOME_AMT`, `SPECIAL_CONSIDERATIONS`, etc.
  - Numerical variables such as `ASK_AMT`.
- Removed Variables
  - `EIN` AND `NAME` -- These were identifiers that do not carry predictive value.
  - Low-frequency classification variables -- Recoded to `"Other"` if they appeared fewer than 50 times in the dataset to reduce noise overfitting.
 
## Compiling, Training, and Evaluating the Model
**Base Model**
- Architecture:
  - Hidden Layers: 3
  - Units: 80 -> 30 -> 1
  - Activations: `relu` -> `relu` -> `sigmoid`
  - Epochs: 100
- Accuracy: `0.731`

**Optimization Attempt 1**
- Architecture:
  - Hidden Layers: 3
  - Units: 124 -> 64 -> 1
  - Activations: `relu` -> `relu` -> `sigmoid`
  - Epochs: 50
- Accuracy: `0.728`

**Optimization Attempt 2**
- Architecture:
  - Hidden Layers: 4
  - Units: 80 -> 30 -> 15 -> 1
  - Activations: `relu` -> `tanh` -> `tanh` -> `sigmoid`
  - Epochs: 50
- Accuracy: `0.729`

**Optimization Attempt 3**
- Architecture:
  - Hidden Layers: 3
  - Units: 80 -> 30 -> 1
  - Activations: `relu` -> `linear` -> `sigmoid`
  - Epochs: 50
- Accuracy: `0.729`

**None of the optimization attempts reached the 75% threshold**

##Optimization Structures
- **Adjusted Binning:** Changed threshold from 50 to 250 for low frequency classification values.
- **Architecture Tuning:**
  - Increased neuron counts in hidden layers.
  - Changed activation functions (`relu`, `tanh`, and `linear`).
  - Added an additional hidden layer.
- **Training Adjustment:** Increased training epochs to 50 for all models.

##Summary
The final deep learning model achieved an accuracy of ~72.9%, falling just short of the target 75% threshold. While multiple architecture and data-related optimizations were attempted, no single change provided a significant improvement in performance.
### Recommendation
Given the marginal returns on increasing model complexity, a different approach may be more effective:
- Try a Gradient Boosting Classifier:
  - Tree-based models often handle tabular data better than deep neural networks.
  - They can better manage categorical variables and imbalanced classes.
  - They offer interpretability through feature importance.
Using a boosting method or even a simple Random Forest may outperform a deep learning mdoel in this context with less tuning and greater interpretability.


