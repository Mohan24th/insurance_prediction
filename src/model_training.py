import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

x_train=pd.read_csv('C:/Users/ram/OneDrive/Desktop/folder/insurance_prediction/dataa/processed/x_train_scaled.csv')
y_train=pd.read_csv('C:/Users/ram/OneDrive/Desktop/folder/insurance_prediction/dataa/processed/y_train.csv')
x_test=pd.read_csv('C:/Users/ram/OneDrive/Desktop/folder/insurance_prediction/dataa/processed/x_test_scaled.csv')
y_test=pd.read_csv('C:/Users/ram/OneDrive/Desktop/folder/insurance_prediction/dataa/processed/y_test.csv')
print(x_train.head())
model=LinearRegression()
model.fit(x_train,y_train)
with open("../artifacts/model.pkl", "wb") as f:
    pickle.dump(model, f)