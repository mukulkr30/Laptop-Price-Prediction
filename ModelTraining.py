import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn import metrics
import joblib
# read data
data = pd.read_csv("Laptop_price.csv")

# PreProcessing the data feature scalling string value encoding and converting it into list
X_notscaled = data[["RAM_Size","Storage_Capacity","Processor_Speed","Screen_Size"]]
brand = data["Brand"]
Y = data["Price"]

#feature scalling of X
scalar = StandardScaler()
X_scaled = scalar.fit_transform(X_notscaled)  #give a numpy array
X_scaled = pd.DataFrame(X_scaled,columns = X_notscaled.columns) #again convert into dataframe

# converting the brand name in relative code
brand = pd.get_dummies(brand,columns = ['Brand'])

#concatenate the X_scaled and brand
X = pd.concat([X_scaled,brand],axis = 1)

# Split into training and testing sets (80:20 ratio)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Convert to lists as requested
X_train_list = np.array(X_train.values.tolist())
X_test_list = np.array(X_test.values.tolist())
Y_train_list = np.array(Y_train.tolist())
Y_test_list = np.array(Y_test.tolist())

# Training the model


model = linear_model.LinearRegression()
model.fit(X_train_list,Y_train_list)

# testing the data set
Y_predict = model.predict(X_test_list)
#print(Y_predict)
r2 = metrics.r2_score(Y_test_list,Y_predict)
mse = metrics.mean_squared_error(Y_test_list,Y_predict)
rmse = np.sqrt(mse)

print(r2)
print(rmse)
# predicting
value = np.array([[8,512,3.8,15.6]])
b = np.array([[0,0,1,0,0]])
scaled_val = scalar.transform(value)
final = np.hstack([scaled_val,b])
ans = model.predict(final)
print(ans*1.055)

joblib.dump(model, "Laptop_price_model.pkl")
joblib.dump(scalar,"Scaling.pkl")