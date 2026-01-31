import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA

class DataProcessor:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.label = LabelEncoder()
        self.pca = PCA(n_components=2)

    def preprocess(self):
        df = self.df.dropna().drop_duplicates()

        if "Charging Start Time" in df.columns:
            df["Charging Start Time"] = pd.to_datetime(df["Charging Start Time"])
            df["Charging End Time"] = pd.to_datetime(df["Charging End Time"])
            df["Duration_Sec"] = (
                df["Charging End Time"] - df["Charging Start Time"]
            ).dt.total_seconds()
            df.drop(columns=["Charging Start Time", "Charging End Time"], inplace=True)

        df.drop(columns=[c for c in ["User ID", "Charging Station ID"] if c in df.columns],
                inplace=True)

        obj_cols = df.select_dtypes(include="object").columns
        for col in obj_cols:
            if df[col].nunique() <= 2:
                df[col] = self.label.fit_transform(df[col])
            else:
                enc = self.encoder.fit_transform(df[[col]])
                enc_df = pd.DataFrame(
                    enc,
                    columns=self.encoder.get_feature_names_out([col]),
                    index=df.index
                )
                df = pd.concat([df, enc_df], axis=1)
                df.drop(col, axis=1, inplace=True)

        scaled = self.scaler.fit_transform(df)
        pca_2d = self.pca.fit_transform(scaled)

        return df, scaled, pca_2d
