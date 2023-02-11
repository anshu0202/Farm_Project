import pandas as pd
df = pd.read_csv('datafile.csv')

df=pd.DataFrame(df)
# print(df)






df1 = pd.read_csv('datafile (1).csv')
df1=pd.DataFrame(df1)
data_top=df1.head()
print(data_top)

# print(df1.to_string())


# print("\n****************\n")
# df2 = pd.read_csv('datafile(2).csv')
# print(df2.to_string())

# print("\n****************\n")
# df3 = pd.read_csv('datafile(3).csv')
# print(df3.to_string())



# print("\n****************\n")
produce = pd.read_csv('produce.csv')


# data_top=produce.head()
# print(data_top)

# produce=pd.DataFrame(produce)

# print(produce.to_string())



# ***************************************
# import pandas as pd
# from sklearn.tree import DecisionTreeClassifier

# # Load the dataset into a pandas DataFrame
# df = pd.read_csv("crops_data.csv")

# # Split the data into features (X) and labels (y)
# X = df.drop("Productivity", axis=1)
# y = df["Productivity"]

# # Train the decision tree model
# model = DecisionTreeClassifier()
# model.fit(X, y)

# # Function to predict productivity based on crop name, location, temperature, humidity, and rainfall
# def increase_productivity(crop_name, location, temperature, humidity, rainfall):
#     data = [[crop_name, location, temperature, humidity, rainfall]]
#     prediction = model.predict(data)
#     return prediction

# # Example Usage
# print(increase_productivity("Rice", "Tropical", 30, 75, 120))




# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder



# df=pd.read_csv("Crop_recommendation.csv")
# # df=pd.DataFrame(df)


# # It is used to know the columns of the Dataset
# # print(df.columns)




# encoder = LabelEncoder()
# df['label'] = encoder.fit_transform(df['label'])




# # Setting the independent and dependent features of the dataset
# features = df.iloc[:, :-1] # all columns except the last one
# target = df.iloc[:, -1] # the last column i.e crop name is the target  here

# encoder = LabelEncoder()
# df['label'] = encoder.fit_transform(df['label'])






# # print("no of rows :", features.shape[0])


# # In this example, the data.data and data.target represent the input features and target variables, respectively. The test_size parameter specifies what percentage of the data should be used for testing (in this case 20%). The random_state parameter is an optional parameter that allows you to specify the random seed, so that you get the same split each time you run the code. This can be useful if you want to reproduce your results.



# X_train,X_test,y_train,y_test= train_test_split(features,target,test_size=0.30, random_state=42)

# # print("Traing dataset is :")
# # print(X_train.shape[0])

# # print("*****************************************************")

# # print("Testing dataset is : ")
# # print(X_test.shape[0])




# # now implementing linear regression


# #standadrizing the dataset
# from sklearn.preprocessing import StandardScaler


# #it is used to initialize scaler
# scaler=StandardScaler()
# # print(X_train)

# X_train=  scaler.fit_transform(X_train)



# # print("*****************************************************")
# # print(X_train)

# X_test=scaler.transform(X_test)



# #to revserse the changes
# # X_train=scaler.inverse.tarnsform(X_train)




# from sklearn.linear_model import LinearRegression

# #cross validation
# from sklearn.model_selection import cross_val_score

# #creating a regression object
# regression=LinearRegression()

# regression.fit(X_train, y_train)

# # cv is used to specify no. of time we have to perform the cross validation i.e 5 models would be created and 

# y_pred = model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# print("Mean Squared Error:", mse)



# # mse=cross_val_score(regression,X_train,y_train,scoring="neg_mean_squared_error",cv=10)


# # print("In progress...")


# # it will give the difference between the predicted and the true value  so its value should be less as possible
# # mse=np.mean(mse)

# # print(mse)
# # print(res)


# # now we will do prediction
# reg_pred=regression.predict(X_test)


# print("predicted data is :")
# # print(reg_pred)


# to konw wheather the predicted value is correct or not we will verify it with the  truth value i.e y_test


# difference=reg_pred-y_test



# plt.plot(difference)
# plt.xlabel('Index')
# plt.ylabel('Difference (True - Predicted)')
# plt.title('Difference between True and Predicted Values')
# # plt.show()




# x = np.radians(reg_pred)
# y = np.sin(x)

# plt.plot(reg_pred, y, label='Predicted values')
# plt.plot(reg_pred, y_test, label='Actual values')
# plt.xlabel('X values (degrees)')
# plt.ylabel('Y values (sine)')
# plt.title('Sine Graph')
# plt.legend()
# plt.show()


# the variance is between +10 to -10 means the model has done good prediction


# from sklearn.metrics import r2_score
# score=r2_score(reg_pred,y_test)

# # it will give adjucted r2 value

# print(score)









# import seaborn as sns

# to get visual representation

#it will give distance plot
# print(sns.displot(reg_pred-y_test))


#Here we have used RandomForest to predict the crop name 





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
print("Accuracy: ", accuracy)


# Create a feature set for a single sample
X_new = [[69,55,38,22.70883798,82.63941394,5.70080568,271.3248604]] # Replace ... with the remaining feature values

# Predict the crop name for the new sample
y_new = clf.predict(X_new)

# The predicted crop name is the first (and only) item in the y_new array
# print("Predicted crop name: ", y_new[0])
def predict_value():
    X_new = [[69,55,38,22.70883798,82.63941394,5.70080568,271.3248604]]
    y_new = clf.predict(X_new)
    return y_new




# import pandas as pd
# df = pd.read_csv('Crop_yeild.csv')

# df=pd.DataFrame(df)
# print(df)







# data_top=df.head()
# print(data_top)




