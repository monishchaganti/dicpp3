#importing liobraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#reading the datset
df = pd.read_csv("data_cardiovascular_risk.csv")

print(df.head())

#making a deep copy of the data frame
df_copy = df.copy()



print(df_copy['glucose'].describe())
print(df_copy['glucose'].isnull().sum())

#filling the null values based on statistical metrics which define how the data is ditributed
df_copy['glucose'] = df_copy['glucose'].fillna(df_copy['glucose'].median())
print(df_copy['glucose'].isnull().sum())

print(df_copy['education'].describe())
print(df_copy['education'].isnull().sum())

plt.hist(df_copy["education"])

df_copy['education'].mode()

df_copy['education'] = df_copy['education'].fillna(df_copy['education'].mode()[0])
print(df_copy['education'].isnull().sum())

print(df_copy['BPMeds'].describe())
print(df_copy['BPMeds'].isnull().sum())

df_copy['BPMeds'].unique()

df_copy['BPMeds'] = df_copy['BPMeds'].fillna(df_copy['BPMeds'].mode()[0])
print(df_copy['BPMeds'].isnull().sum())

print(df_copy['totChol'].describe())
print(df_copy['totChol'].isnull().sum())

df_copy['totChol'] = df_copy['totChol'].fillna(df_copy['totChol'].median())
print(df_copy['totChol'].isnull().sum())

print(df_copy['cigsPerDay'].describe())
print(df_copy['cigsPerDay'].isnull().sum())

df_copy['cigsPerDay'] = df_copy['cigsPerDay'].fillna(df_copy['cigsPerDay'].median())
print(df_copy['cigsPerDay'].isnull().sum())

print(df_copy['BMI'].describe())
print(df_copy['BMI'].isnull().sum())

df_copy['BMI'] = df_copy['BMI'].fillna(df_copy['BMI'].median())
print(df_copy['BMI'].isnull().sum())

print(df_copy['heartRate'].describe())
print(df_copy['heartRate'].isnull().sum())

df_copy['heartRate'] = df_copy['heartRate'].fillna(df_copy['heartRate'].median())
print(df_copy['heartRate'].isnull().sum())

#label encoding the categorical data into numerical data
le=LabelEncoder()
df_copy['sex']=le.fit_transform(df_copy['sex'])
df_copy['is_smoking']=le.fit_transform(df_copy['is_smoking'])

df_copy.head()


#creating a new column BP_Category from the existing columns based on sysBP and diaBP 
def categorize_blood_pressure(row):
    sys_bp = row['sysBP']
    dia_bp = row['diaBP']
    
    if sys_bp < 120 and dia_bp < 80:
        return 'Normotension'
    elif 120 <= sys_bp <= 139 or 80 <= dia_bp <= 89:
        return 'Prehypertension'
    elif 140 <= sys_bp <= 159 or 90 <= dia_bp <= 99:
        return 'Stage 1 Hypertension'
    else:
        return 'Stage 2 Hypertension'

# Applying the categorization function
df_copy['BP_Category'] = df_copy.apply(categorize_blood_pressure, axis=1)
print(df_copy)

#Dropping the column is_smolking from data frame to reduce dimensionality
df_copy.drop('is_smoking', axis = 1, inplace = True)

#creating a new column BMI_Category from the existing column based on BMI
bmi_bins = [0, 18.5, 24.9, float('inf')]
bmi_labels = ['Underweight', 'Normal', 'Overweight']

# Create a new column 'BMI_Category' based on the defined bins and labels
df_copy['BMI_Category'] = pd.cut(df_copy['BMI'], bins=bmi_bins, labels=bmi_labels, right=False)

# Display the DataFrame with the new 'BMI_Category' column
print(df_copy)

#checking for any moree null values
print(df_copy.isnull().sum())

# EXPLORATORY DATA ANALYSIS

plt.hist(df_copy["TenYearCHD"])

#BP_Category distribution
plt.hist(df_copy["BP_Category"])

#BMI_Category distribution
plt.hist(df_copy["BMI_Category"])


#dropping id column
df_copy.drop(['id'],axis=1,inplace=True) #Id is not useful for Model training.

#creating a new daaframe with only numerical values
dfnum = df_copy.drop(columns=['BP_Category','BMI_Category']).copy()


#splitting the data into training and testing
X = dfnum.iloc[:,:-1]
y = dfnum['TenYearCHD']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set shape (X, y):", X_train.shape, y_train.shape)
print("Testing set shape (X, y):", X_test.shape, y_test.shape)

print(X_train.isnull().sum())
print(y_train.isnull().sum())

#standardizing the data
features_to_standardize = X_train.columns
scaler = StandardScaler()
df_std_train = pd.DataFrame(scaler.fit_transform(X_train[features_to_standardize]), columns=features_to_standardize)

df_std_test = pd.DataFrame(scaler.fit_transform(X_test[features_to_standardize]), columns=features_to_standardize)

from sklearn.preprocessing import MinMaxScaler

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform the testing data (using the parameters learned from the training data)
X_test_scaled = scaler.transform(X_test)

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

print(X_train_scaled_df.isnull().sum())
print(X_test_scaled_df.isnull().sum())

# Model Building


# Install imbalanced-learn using pip
import subprocess

# Run the pip install command
subprocess.call(["pip", "install", "imbalanced-learn"])


from imblearn.over_sampling import SMOTE
from collections import Counter

# Check class distribution before SMOTE
print("Class distribution before SMOTE:", Counter(y_train))

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled_df, y_train)

# Check class distribution after SMOTE
print("Class distribution after SMOTE:", Counter(y_train_resampled))


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report



from sklearn.ensemble import GradientBoostingClassifier

# Initialize Gradient Boosting classifier
gb = GradientBoostingClassifier()

# Train the model
gb.fit(X_train_resampled, y_train_resampled)

# Predict on the test set
y_pred_gb = gb.predict(X_test_scaled_df)


# Evaluate Gradient Boosting model
accuracy_gb = accuracy_score(y_test, y_pred_gb)
print("Accuracy Score (Gradient Boosting):", accuracy_gb)

# Classification report for Gradient Boosting
print("Classification Report (Gradient Boosting):\n", classification_report(y_test, y_pred_gb))



# import pickle

# with open('model.pkl', 'wb') as f:
#     pickle.dump(gb, f)

import pickle
from sklearn.pipeline import Pipeline

# Assuming `gb` is your GradientBoostingClassifier and `scaler` is your MinMaxScaler

# Create a Pipeline with the scaler and the classifier
pipeline = Pipeline([
    ('scaler', scaler),  # Assuming you've named your MinMaxScaler object as `scaler`
    ('classifier', gb)   # Assuming you've named your GradientBoostingClassifier object as `gb`
])

# Save the pipeline (including both scaler and classifier) to a file
with open('model_with_scaler.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
