Final Project: Reservoir Production Forecasting Using Deep Neural Networks
Project Overview
This project focuses on developing a deep learning-based proxy model to forecast oil production from hydrocarbon reservoirs. The model utilizes spatial data from permeability and porosity maps (64x64), along with a scalar value for initial water saturation, to predict 12 distinct target variables representing oil rate and cumulative production over six different time steps.

Methodology and Key Findings on the Initial Dataset (756 Samples)
The primary development and optimization of the model were conducted on an initial dataset containing 756 unique reservoir samples. Two main approaches were investigated:

Model Trained on All Data: In this primary approach, the model was trained on the complete set of 756 samples, including statistical outliers. After extensive hyperparameter tuning using the Optuna framework, this model achieved an excellent final accuracy with an R² score of approximately 97.7% on the test set. This was determined to be the most robust and reliable approach.

Model Trained After Outlier Removal: As a comparative experiment, a side project was conducted where statistical outliers were first removed from the dataset using the IQR method. A new model was then trained and optimized on this "cleaned" dataset, achieving a final accuracy with an R² score of 93.3%.

Important Note on the Dataset
After the completion of the main analysis, it was discovered that the initial project dataset was an incomplete version. The full dataset, as originally intended, contains 1050 samples. The initial work was based on 756 samples due to an incomplete file download.

Late Project Submission (late project folder)
Due to the time constraints at the end of the project, it was not feasible to re-run the entire, comprehensive optimization process (such as the extensive Optuna search) on the full 1050-sample dataset with an R² score of 91.3%.

However, an initial attempt was made to train the model on the larger, complete dataset. The code and preliminary results of this effort are located in the late project folder. It is important to note that because this version of the model did not undergo the same rigorous fine-tuning, its results do not match the high accuracy achieved by the final models developed on the 756-sample dataset.


Created by: $$Novin Nekuee | Soroush Danesh $$