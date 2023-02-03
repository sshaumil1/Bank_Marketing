import numpy as np
import json,pickle,config

class TermDeposit():
    def __init__(self,age,education,default,balance,housing,loan,duration,campaign,pdays,previous,job):
        self.age        = age
        self.education  = education
        self.default    = default
        self.balance    = balance
        self.housing    = housing
        self.loan       = loan
        self.duration   = duration
        self.campaign   = campaign
        self.pdays      = pdays
        self.previous   = previous
        self.job        = job
    def get_models(self):
        with open (config.JSON_FILE_PATH, "r") as f:
            self.json_data = json.load(f)
        with open (config.MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)
        with open (config.SCALING_PATH, "rb") as f:
            self.std = pickle.load(f)
    def get_prediction(self):
        self.get_models()
        self.cols = list(self.json_data["columns"])
        test_arr = np.zeros(len(self.cols))
    
        test_arr[0] = self.age
        test_arr[1] = self.json_data["education"][self.education]
        test_arr[2] = self.json_data["default"][self.default]
        test_arr[3] = self.balance
        test_arr[4] = self.json_data["housing"][self.housing]
        test_arr[5] = self.json_data["loan"][self.loan]
        test_arr[6] = self.duration
        test_arr[7] = self.campaign
        test_arr[8] = self.pdays
        test_arr[9] = self.previous
        index       = self.cols.index(self.job)
        test_arr[index] = 1

        scalled_data = self.std.transform([test_arr])
        predicted_class = self.model.predict(scalled_data)
        return predicted_class

