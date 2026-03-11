import pickle
import numpy as np


class InsurancePremiumPredictor:

    def __init__(self):

        with open("artifacts/scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)

        with open("artifacts/model.pkl", "rb") as f:
            self.model = pickle.load(f)

    def predict(self, Age, Annual_Income_LPA, Policy_Term_Years, Sum_Assured_Lakhs):

        input_data = np.array([[Age, Annual_Income_LPA, Policy_Term_Years, Sum_Assured_Lakhs]])

        scaled_input = self.scaler.transform(input_data)

        prediction = self.model.predict(scaled_input)

        return prediction[0]