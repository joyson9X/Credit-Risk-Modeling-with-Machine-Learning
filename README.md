# Credit-Risk-Modeling-with-Machine-Learning

# Project Objective
This project involves building a machine learning model to predict the probability of a borrower defaulting on a loan. The primary goal is to move beyond traditional analysis and use a data-driven approach to identify high-risk applicants, thereby minimizing potential financial losses for a lending institution.

The entire workflow, from data exploration and preprocessing to model training, evaluation, and business impact analysis, is documented in the accompanying Jupyter Notebook.

# Key Features & Methodology
The project follows a systematic data science workflow, beginning with a thorough Exploratory Data Analysis (EDA). This initial phase involved investigating a large-scale dataset of over 2 million loans to identify key patterns, feature distributions, correlations, and the significant class imbalance between defaulted and fully paid loans.

Following the EDA, the next phase was crucial Data Preprocessing. This involved selecting a targeted subset of the most impactful features based on the initial analysis, handling missing values, and converting categorical variables into a numerical format using one-hot encoding to create a model-ready dataset.

With a clean dataset, the core of the project was Model Training. A Logistic Regression model was trained on a sample of the data, incorporating key techniques such as Feature Scaling to standardize the data for improved performance and Handling Class Imbalance by using the class_weight='balanced' parameter to ensure the model gives appropriate importance to the minority class (defaults).

To ensure reliability, the model was subjected to a rigorous Model Evaluation on an unseen test set using industry-standard metrics, including the AUC-ROC curve, a detailed classification report, and a confusion matrix.

Finally, the project culminates in a Business Impact Analysis, where the model's predictive performance is translated into a tangible financial outcome, quantifying the potential loss reduction a bank could achieve by implementing this model.

# Key Findings & Results
The model demonstrated strong predictive power with significant business value:

High Recall: The model successfully identified 69% of all actual loan defaults in the unseen test data.

Strong AUC Score: Achieved an AUC-ROC score of 0.68, indicating a good ability to distinguish between defaulting and non-defaulting borrowers (where 0.5 is random chance).

Actionable Business Impact: The analysis concluded that by declining just the top 10% of applicants identified as riskiest by the model, a lender could prevent over 25% of its total potential default losses.

# How to Run This Project
1. Get the Data:
The dataset used (accepted_2007_to_2018Q4.csv.gz) is too large to be hosted on GitHub. Please download it from its official source on Kaggle: Lending Club Loan Data on Kaggle. Once downloaded, place the .csv.gz file in the same directory as the Jupyter Notebook.

2. Setup the Environment:
This project uses Python. The required libraries can be installed by running the following command in your terminal or the first cell of the notebook: pip install pandas matplotlib seaborn scikit-learn

3. Run the Notebook:
Open the .ipynb file in Jupyter Lab or Jupyter Notebook. Make sure the file_path variable in the first code cell points to the correct location of your downloaded data file. Run the cells sequentially from top to bottom to replicate the analysis.

Technology Stack
Language: Python
Libraries:
Data Analysis: Pandas
Data Visualization: Matplotlib, Seaborn
Machine Learning: Scikit-learn
