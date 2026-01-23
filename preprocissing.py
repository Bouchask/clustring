from sklearn.preprocessing import StandardScaler,OneHotEncoder
import numpy as np 
import pandas as pd 
class Data :
    def __init__(self , url : str ):
        self.data = pd.read_csv(url)
        self.data = pd.DataFrame(self.data)
        self.scaler = StandardScaler()
        self.oh = OneHotEncoder(handle_unknown="ignore")
    def clean_data(self,data):
        data = data.dropna()
        data = data.drop_duplicates()
        return data 
    def Scaler_data(self,data,list_feature_number):
        data[list_feature_number] = self.scaler.fit_transform(data[list_feature_number])
        return data
    def label_data(self,data,list_feature_object):
        for i in list_feature_object :
            data = self.oh.fit_transform(data[i])
            data.drop(i,axis=1,inplace=True)
        return data 
    def dateTo_Second(self,data,list_column_date): #date start - date End
        data[list_column_date[1]] = pd.to_datetime(data[list_column_date[1]])
        data[list_column_date[0]] = pd.to_datetime(data[list_column_date[0]])
        data["Time_second"] = (data[list_column_date[1]] - data[list_column_date[0]]).dt.total_seconds()
        data.drop(labels=list_column_date,axis=1,inplace=True)
        return data
    def preprocessing(self):
        data = self.data.copy()
        data = self.clean_data(data)
        data.drop("User ID",axis = 1,inplace=True)
        data = self.dateTo_Second(data,["Charging Start Time","Charging End Time"])
        list_feature_object = data.select_dtypes(include="object").columns
        data = self.label_data(data,list_feature_object)
        list_column_number = [name_column for name_column in data.columns if not  (776>= (data[name_column].nunique()) >=2 )]
        data = self.Scaler_data(data,list_column_number)

