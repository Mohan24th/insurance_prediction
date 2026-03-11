#load Traning and testing data
# scale the trraning data
# save scaled data into processed folder

from data_preprocessing import Load_and_split_data
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

x_train,x_test,y_train,y_test=Load_and_split_data()

scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)

pd.DataFrame(x_train_scaled, columns=x_train.columns).to_csv('C:/Users/ram/OneDrive/Desktop/folder/insurance_prediction/dataa/processed/x_train_scaled.csv', index=False)
pd.DataFrame(x_test_scaled, columns=x_test.columns).to_csv('C:/Users/ram/OneDrive/Desktop/folder/insurance_prediction/dataa/processed/x_test_scaled.csv', index=False)
pd.DataFrame(y_train).to_csv('C:/Users/ram/OneDrive/Desktop/folder/insurance_prediction/dataa/processed/y_train.csv', index=False)
pd.DataFrame(y_test).to_csv('C:/Users/ram/OneDrive/Desktop/folder/insurance_prediction/dataa/processed/y_test.csv', index=False)    
with open("../artifacts/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
    
