import numpy as np
import pandas as pd

class Preprocessor:
    def __init__(self, df):
        self.df = df
    
    def preprocess(self):
        categorical_features = self.df.iloc[:, :1]
        categorical_features_processed = self.preprocess_categorical(categorical_features)

        numeric_features = self.df.iloc[:, 1:]
        numeric_features_processed = self.preprocess_numeric(numeric_features)

        processed_df = pd.concat([pd.DataFrame(categorical_features_processed), numeric_features_processed], axis=1)
        return processed_df
    
    def preprocess_numeric(self, numeric_data):
        imputed_numeric_data = self.numeric_imputation(numeric_data)
        removed_outliers_numeric_data = self.remove_outliers(imputed_numeric_data)
        imputed_numeric_data = self.numeric_imputation(removed_outliers_numeric_data)
        standardized_numeric_data = self.standardization(imputed_numeric_data)

        return standardized_numeric_data
    
    def preprocess_categorical(self, categorical_data):
        categorical_data = self.categorical_imputation(categorical_data)
        
        return categorical_data

    def numeric_imputation(self, data):
        for col in data.columns:
            col_mean = data[col].mean()
            col_median = data[col].median()
            data.fillna({col: col_mean}, inplace=True)

        return data
    
    def remove_outliers(self, data):
        for col in data.columns:
            col_z_scores = (data[col] - data[col].mean()) / data[col].std()
            data.loc[col_z_scores.abs() > 3.5, col] = np.nan
        
        return data
    
    def standardization(self, data):
        for col in data.columns:
            col_mean = data[col].mean()
            col_std = data[col].std()
            if col_std != 0:
                data[col] = (data[col] - col_mean) / col_std
            else:
                data[col] = 0
        
        return data
    
    def categorical_imputation(self, data):
        for col in data.columns:
            mode = data[col].mode()[0]
            data[col] = data[col].fillna(mode)

        return data