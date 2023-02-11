
# This is the ML module

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data into a pandas dataframe
df = pd.read_csv("Crop_recommendation.csv")

# Divide the data into features (X) and target (y)
# X = df.drop("crop_name", axis=1)
# y = df["crop_name"]


# # Setting the independent and dependent features of the dataset
X= df.iloc[:, :-1] # all columns except the last one
y= df.iloc[:, -1] # the last column i.e crop name is the target  here




# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy Of ML module : ", accuracy)


# Create a feature set for a single sample
# Later on this data would be replaced by the data as per the location given by the user
# X_new = [[17,59,17,18.416700100000003,23.42829938,5.689858133,132.9801054]] # Replace ... with the remaining feature values

# Predict the crop name for the new sample based on the user location provided
# y_new = clf.predict(X_new)

# The predicted crop name is the first (and only) item in the y_new array
# print("Predicted crop name: ", y_new[0])

# print("Predicted crop : ",y_new)

def predict_value():
    X_new = [[69,55,38,22.70883798,82.63941394,5.70080568,271.3248604]]
    y_new = clf.predict(X_new)
    return y_new

