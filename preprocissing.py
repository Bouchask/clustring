from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
from sklearn.decomposition import PCA
import numpy as np 
import pandas as pd 
class Data :
    def __init__(self , url : str ):
        self.data = pd.read_csv(url)
        self.data = pd.DataFrame(self.data)
        self.scaler = StandardScaler()
        self.oh = OneHotEncoder(handle_unknown="ignore")
        self.lc = LabelEncoder()
        self.pca = PCA(n_components=2)
    def clean_data(self,data):
        data = data.dropna()
        data = data.drop_duplicates()
        return data 
    def Scaler_data(self,data,list_feature_number):
        data[list_feature_number] = self.scaler.fit_transform(data[list_feature_number])
        pca_data = self.pca.fit_transform(data[list_feature_number])
        return data
    def pca_data(self , data , list_feature_number ):
        pca_data = self.pca.fit_transform(data[list_feature_number])
        pca_data = pd.DataFrame(pca_data,columns = ["pca1","pca2"],index = data.index)
        data.drop(list_feature_number , axis = 1 , inplace = True)
        data = pd.concat([data,pca_data],axis = 1)
        return data
    def label_data(self,data,list_feature_object):
        label_code = self.oh.fit_transform(data[list_feature_object])
        label_code = pd.DataFrame(label_code.toarray(),columns = self.oh.get_feature_names_out(list_feature_object),index = data.index)
        data.drop(list_feature_object , axis = 1 , inplace = True)
        data = pd.concat([data,label_code],axis = 1 )
        return data 
    def binaire_feature(self,data , list_feature):
        for i in list_feature :
            data[i] = self.lc.fit_transform(data[i])
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
        
        list_column_number = [name_column for name_column in data.columns if (776 < (data[name_column].nunique()))]
        list_feature_binaire = [name_column for name_column in list_feature_object if ((data[name_column].nunique()) == 2 )]
        list_feature_ulticlasses =  [name_column for name_column in list_feature_object if (776>= (data[name_column].nunique()) >2 )]
        data = self.label_data(data,list_feature_ulticlasses)
        data = self.binaire_feature(data,list_feature_binaire)
        data = self.Scaler_data(data,list_column_number)
        data = self.pca_data(data , list_column_number)
        return data
