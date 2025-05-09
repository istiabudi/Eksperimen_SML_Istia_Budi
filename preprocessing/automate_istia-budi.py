import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


def preprocess_data(file_path):

    data = pd.read_csv(file_path)

    nums = ['Temperature', 'Humidity', 'SquareFootage', 'Occupancy', 'RenewableEnergy']
    numeric_feature = data[nums]
    cat = ['Month', 'Hour', 'Occupancy', 'DayOfWeek', 'Holiday', 'HVACUsage', 'LightingUsage']
    cat_feature = data[cat]
    label = data['EnergyConsumption']

    scaler = StandardScaler()
    num_scaled = scaler.fit_transform(numeric_feature)
    num_scaled_df = pd.DataFrame(num_scaled, columns=nums)
    
    for col in cat_feature.columns:
        le = LabelEncoder()
        cat_feature[col] = le.fit_transform(cat_feature[col])

    cat_feature_df = pd.DataFrame(cat_feature, columns=cat)

    data = pd.concat([num_scaled_df, cat_feature_df, label], axis=1)
    print('selesai')
    
    return data.to_csv('preprocessing/energy_preprocessed.csv', index=False)

if __name__ == "__main__":
    preprocess_data('./energy_dataset.csv')
    