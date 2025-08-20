from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

class DataPreprocessor:
    def __init__(self, data):
        self.data = data

    def downsample(self, fraction):
        num_rows_to_keep = int(len(self.data) * fraction)

        # Downsample the data by randomly selecting rows
        downsampled_data = self.data.sample(n=num_rows_to_keep)
        self.data = downsampled_data

        return self.data

    def label_encode(self):
        class_list=[]
        for i in range (len(self.data)):
            element=self.data[i]
            last_column_values = element['class'].tolist()
            class_list.extend(last_column_values)
        unique_values_set = set(class_list)
        unique_values_list = list(unique_values_set)
        # Perform label encoding on the categories
        label_encoder = LabelEncoder()
        encoded_data = label_encoder.fit_transform(unique_values_list)

        label_mapping = {unique_values_list[i]: encoded_data[i] for i in range(len(unique_values_list))}
        encoded_categories = list(label_encoder.classes_)
        actual_values = list(encoded_data)
        for i in range(len(encoded_categories)):
            print(f'{encoded_categories[i]}: {actual_values[i]}')
        return label_mapping
        
    def normalize(self):
        # Perform data normalization
        scaler = MinMaxScaler(feature_range=(-1,1))
        features_to_normalize = ['xCenter','yCenter','xVelocity','yVelocity','heading','xAcceleration','yAcceleration']
        scaler.fit(self.data[features_to_normalize])
        self.data[features_to_normalize] = scaler.transform(self.data[features_to_normalize])
        return self.data

# PS: Recommended to return the processed data to the main.ipynb for further application