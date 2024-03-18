# Advanced_Machine_Learning_project_1

Datasets (small, need 3, max. variables):
- Heart disease from UCI repository, predicting  fasting blood sugar > 120 mg/dl https://archive.ics.uci.edu/dataset/45/heart+disease
- 7 variables + target Apple quality https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality
- Fertility txt file, but well described:  https://archive.ics.uci.edu/dataset/244/fertility
- ARFF format, 7 variables we might skip this one:  rice classification https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik

Datasets (big, need 6, min. variables):
- Breast cancer from UCI repository Diagnosis (M(1) = malignant, B(0) = benign) https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- 25 variables, some to binarize, https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction
- 25 variables,  water-quality https://www.kaggle.com/datasets/mssmartypants/water-quality
- 12 variables for wine dataset, oklepane ale można złączyć czerwone i białe wina i wtedy miec extra target  https://archive.ics.uci.edu/dataset/186/wine+quality
- 36 Features, https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
- Algerian Forest Fire:  (to be checked)   https://archive.ics.uci.edu/dataset/547/algerian+forest+fires+dataset
- 60 features, Mine vs. Rock https://www.openml.org/search?type=data&sort=runs&id=40&status=active


# VENV ACTIVATION
Navigate to the project directory and create a virtual environment: python -m venv <env_name> Note: On some systems, you might need to use python3 instead of python

Step 3: Activate the Virtual Environment
For Windows: <env_name>\Scripts\activate
For macOS and Linux: source <env_name>/bin/activate
Your command prompt or terminal should now show the virtual environment name.

Step 4: Install Dependencies
Install the required dependencies listed in the requirements.txt file using : make install_requirements
