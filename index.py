import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder



df=pd.read_csv("Crop_recommendation.csv")
# df=pd.DataFrame(df)


# It is used to know the columns of the Dataset
# print(df.columns)




encoder = LabelEncoder()
df['label'] = encoder.fit_transform(df['label'])




# Setting the independent and dependent features of the dataset
features = df.iloc[:, :-1] # all columns except the last one
target = df.iloc[:, -1] # the last column i.e crop name is the target  here

encoder = LabelEncoder()
df['label'] = encoder.fit_transform(df['label'])






# print("no of rows :", features.shape[0])


# In this example, the data.data and data.target represent the input features and target variables, respectively. The test_size parameter specifies what percentage of the data should be used for testing (in this case 20%). The random_state parameter is an optional parameter that allows you to specify the random seed, so that you get the same split each time you run the code. This can be useful if you want to reproduce your results.



X_train,X_test,y_train,y_test= train_test_split(features,target,test_size=0.30, random_state=42)

# print("Traing dataset is :")
# print(X_train.shape[0])

# print("*****************************************************")

# print("Testing dataset is : ")
# print(X_test.shape[0])




# now implementing linear regression


#standadrizing the dataset
from sklearn.preprocessing import StandardScaler


#it is used to initialize scaler
scaler=StandardScaler()
# print(X_train)

X_train=  scaler.fit_transform(X_train)



# print("*****************************************************")
# print(X_train)

X_test=scaler.transform(X_test)



#to revserse the changes
# X_train=scaler.inverse.tarnsform(X_train)




from sklearn.linear_model import LinearRegression

#cross validation
from sklearn.model_selection import cross_val_score

#creating a regression object
regression=LinearRegression()

regression.fit(X_train, y_train)

# cv is used to specify no. of time we have to perform the cross validation i.e 5 models would be created and 

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)



mse=cross_val_score(regression,X_train,y_train,scoring="neg_mean_squared_error",cv=10)


# print("In progress...")


# it will give the difference between the predicted and the true value  so its value should be less as possible
mse=np.mean(mse)

# print(mse)
# print(res)


# now we will do prediction
reg_pred=regression.predict(X_test)


print("predicted data is :")
# print(reg_pred)


# to konw wheather the predicted value is correct or not we will verify it with the  truth value i.e y_test


# difference=reg_pred-y_test



# plt.plot(difference)
# plt.xlabel('Index')
# plt.ylabel('Difference (True - Predicted)')
# plt.title('Difference between True and Predicted Values')
# # plt.show()




x = np.radians(reg_pred)
y = np.sin(x)

plt.plot(reg_pred, y, label='Predicted values')
plt.plot(reg_pred, y_test, label='Actual values')
plt.xlabel('X values (degrees)')
plt.ylabel('Y values (sine)')
plt.title('Sine Graph')
plt.legend()
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


# print("helo")




