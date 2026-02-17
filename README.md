# Heart Disease Prediction

This project analyzes health indicators to predict the presence of heart disease. It demonstrates a complete data science workflow: data cleaning, exploratory analysis, feature engineering, handling class imbalance, and building machine learning models.

The final model (Logistic Regression with balanced class weights) achieves reasonable recall, identifying about 42% of actual heart disease cases, while maintaining interpretability.

## 📁 Dataset

- **Source**: [Kaggle - Heart Disease by Oktay Ördekçi](https://www.kaggle.com/datasets/oktayrdeki/heart-disease)
- **Description**: Contains 21 health-related features such as age, cholesterol, exercise habits, stress level, and a binary target indicating heart disease status.
- **Size**: 10,000 rows, 21 columns
- **Target Distribution**: 80% No Heart Disease, 20% Heart Disease (imbalanced)

## 🗂️ Project Structure
heart-disease-analysis/
│
├── data/ # Raw dataset (heart_disease.csv)
├── notebooks/ # Jupyter notebook with full analysis
├── results/ # Outputs: models, scaler, visualizations, summary
├── .gitignore # Files to ignore
├── README.md # Project overview (this file)
├── requirements.txt # Python dependencies
└── LICENSE # MIT License

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/prajwal859/heart-disease-analysis.git
   cd heart-disease-analysis

2. Create and activate a virtual environment:

python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

3. Install required packages:

pip install -r requirements.txt

## Usage

Run the Jupyter notebook for a step-by-step walkthrough:

jupyter notebook notebooks/heart_disease_analysis.ipynb

Or run the Python script directly:
python src/analysis.py

## Results

Exploratory Data Analysis
Target Distribution – The dataset is imbalanced, with only 20% positive cases. This required special handling during modeling.

https://results/target_distribution.png

Feature Importance – Using a Random Forest model (with balanced weights), we identified the top predictors. Note that even though the model did not perform well, the importance scores reflect underlying patterns.

https://results/feature_importance.png

## Model Performance

We trained a Logistic Regression model with class_weight='balanced' to account for the imbalance. The final model uses a decision threshold of 0.5.

Metric	            Value
Accuracy	        51%
Precision (class 1)	18%
Recall (class 1)	43%
F1-score (class 1)	0.26
ROC-AUC	            0.47

Confusion Matrix (on test set, threshold 0.5):

	            Predicted No	Predicted Yes
Actual No	    844	            756
Actual Yes	    230	            170

The model catches 170 out of 400 heart disease cases (43% recall), but at the cost of many false positives (756).

Threshold Tuning
We experimented with different thresholds to balance recall and precision. Lowering the threshold increases recall but drastically reduces accuracy and precision.


Threshold	Accuracy	Precision	Recall	F1-score
0.2	        0.20	    0.20	    1.00	0.33
0.3	        0.20	    0.20	    1.00	0.33
0.4	        0.20	    0.20	    0.998	0.33
0.5	        0.51	    0.18	    0.43	0.26
0.6	        0.80	    0.50	    0.003	0.005

## Limitations
The overall predictive power is modest, likely because the dataset lacks strong signals or contains noisy/synthetic features.

Precision for the positive class is low, meaning many false positives – the model may not be reliable for clinical use.

Class imbalance remains a challenge, and the chosen threshold represents a compromise.

## Future Work
Collect additional features (e.g., family history, genetic markers, detailed lifestyle data).

Try more advanced techniques like SMOTE (synthetic oversampling) or XGBoost with scale_pos_weight.

Gather more data, especially for the minority class.

Perform feature engineering to create new, more predictive variables.

Experiment with different evaluation metrics and cost-sensitive learning.

## Licence
This project is licensed under the MIT License – see the LICENSE file for details.

## Acknowledgements
Dataset provided by Oktay Ördekçi on Kaggle.

Inspired by the UCI Heart Disease dataset.