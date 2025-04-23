## Project Documentation: System Threat Forecaster

### Project Overview

**Project Title**: System Threat Forecaster\
**Course**: Machine Learning Project, BS in Data Science, IIT Madras\
**Grade Achieved**: S Grade (Highest)\
**Rank**: 26 amoung 1700+ participants (based on accuracy score)

**Objective**:\
The aim of this machine learning project is to predict the likelihood of systems getting infected by various malware families using telemetry data collected by antivirus software. The problem is formulated as a binary classification task where the target variable indicates whether a machine is infected (1) or not (0).

### Dataset Description

The dataset comprises telemetry information per machine. The training data (`train.csv`) includes a target column, while the test data (`test.csv`) does not. The features cover antivirus configuration, system specifications, OS details, geographical identifiers, and more.

**Key Files**:

- `train.csv`: Training data with labeled `target`
- `test.csv`: Testing data without labels
- `sample_submission.csv`: Template for submission format

**Evaluation Metric**:

- `accuracy_score()` between predicted classes and ground truth labels

### Feature Description (Not Exhaustive - Check Code For Details)

- `MachineID`: Unique ID for each machine
- `ProductName`, `EngineVersion`, `AppVersion`, `SignatureVersion`: Antivirus product metadata
- `IsBetaUser`, `IsPassiveModeEnabled`, `RealTimeProtectionState`: Antivirus usage details
- `HasTpm`, `FirewallEnabled`, `IsSystemProtected`: System protection status
- `OSVersion`, `OSBuildNumber`, `OSArchitecture`: OS-specific metadata
- `CountryID`, `CityID`, `GeoRegionID`: Geographic info
- `PrimaryDiskCapacityMB`, `TotalPhysicalRAMMB`: Hardware specifications
- `IsVirtualDevice`, `IsTouchEnabled`, `IsAlwaysOnAlwaysConnectedCapable`: Device features
- `DateAS`, `DateOS`: Temporal data for malware signatures and OS updates
- `target`: Binary classification label (1 = infected, 0 = not infected)

---

## Methodology

### 1. **Exploratory Data Analysis (EDA)**

- **Data Dimensions & Types**: Verified column types, missing values, and value distributions.
- **Class Distribution**: Checked imbalance in the `target` variable.
- **Univariate Analysis**: Histograms, bar charts for categorical variables.
- **Bivariate Analysis**: Relation between categorical variables and target using grouped bar plots.
- **Date Features**: Converted `DateAS` and `DateOS` to datetime objects and calculated gaps and recency-based features.

### 2. **Data Preprocessing & Feature Engineering**

- **Handling Missing Values**:
  - Mode imputation for categorical values.
  - Median imputation for numerical features.
- **Feature Transformation**:
  - Categorical Encoding using Label Encoding for tree-based models.
  - Date features converted to number of days since minimum date in the column.
  - Engineered new features from date differences (`days_since_as`, `days_since_os_update`).
- **Feature Cleaning**:
  - Removed features with more than 90% missing or single value across rows.

### 3. **Feature Selection & Dimensionality Reduction**

- **Variance Threshold**: Removed features with low variance.
- **Correlation Matrix**: Dropped highly correlated variables to reduce redundancy.
- **Tree-based Feature Importance**: Used model feature importances (XGBoost/LightGBM) to rank top features.

### 4. **Model Building & Evaluation**

- **Models Tried**:
  - LightGBM
  - XGBoost
  - Random Forest
- **Train-Test Splitting**:
  - Stratified K-Fold Cross Validation (5-fold) for stable performance estimation.
- **Performance Metric**: Accuracy Score

**Best Performing Model**:

- **Model**: Voting Classifier combining LightGBM and HistGradientBoostingClassifier (HistGBM)
- **Approach**: A soft voting ensemble of LightGBM and HistGBM was used to improve generalization.
- **Cross-validation Accuracy**: Highest among all tried models

### 5. **Hyperparameter Tuning**

- **Method**: Grid Search and Randomized Search on LightGBM and HistGradientBoosting
- **Parameters Tuned**:
  - `num_leaves`, `min_samples_leaf`
  - `max_depth`, `max_bins`, `max_iter`
  - `learning_rate`, `l2_regularization`
  - `n_estimators`
  - `subsample`, `colsample_bytree`
- **Early Stopping**: Used validation set to prevent overfitting

---

## Key Insights

- **System protection features** (`IsSystemProtected`, `RealTimeProtectionState`, `FirewallEnabled`) were strong predictors of infection.
- **Antivirus metadata** like `SignatureVersion`, `EngineVersion` correlated well with `target`.
- **Time Gap** between signature date and OS update was significant—machines with long update gaps were more vulnerable.
- **Hardware specs** (RAM, Disk capacity) contributed marginally but helped in feature interaction.

---

## Final Results

- **Final Model**: Voting Classifier with LightGBM and HistGBM (soft voting)
- **Test Set Prediction**: Submitted on Kaggle-styled platform using the given template
- **Final Accuracy Score**: 0.63820 ; ranked **26th overall**

---

## Conclusion

This project covered the full machine learning workflow from data cleaning to hyperparameter tuning. A high degree of attention was given to domain-specific features like system protection and signature update recency. The accuracy metric and cross-validation ensured reliable model selection. The project earned an S grade and performed in the top 30, showcasing both technical and analytical strengths.

---

## Future Improvements

- Explore ensemble models like stacking
- Incorporate interaction features via polynomial or target encoding
- Use anomaly detection as pre-step to identify compromised systems

---

## Acknowledgements

- Faculty and TAs of the Machine Learning course, BS in Data Science, IIT Madras
- Open-source community (scikit-learn, LightGBM, XGBoost, pandas, matplotlib, seaborn, scipy, etc.)

