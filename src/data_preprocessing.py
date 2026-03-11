import pandas as pd
from sklearn.model_selection import train_test_split
def Load_and_split_data():
    data = pd.read_csv('C:/Users/ram/OneDrive/Desktop/folder/insurance_prediction/dataa/raw/insurance.csv')
    print(data.head()) 
    x=data[['Age','Annual_Income_LPA','Policy_Term_Years','Sum_Assured_Lakhs']]
    y=data['Annual_Premium_Thousands']
    print(x)
    print(y)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
    return x_train, x_test, y_train, y_test

