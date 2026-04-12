import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def one_hot_encode(data, categorical_features):
    encoder = OneHotEncoder(sparse_output=False)
    encoded_data = encoder.fit_transform(data[categorical_features])
    return encoded_data

# Example usage
# data = pd.DataFrame({
#     'Color': ['Red', 'Green', 'Blue', 'Green'],
#     'Size': ['S', 'M', 'L', 'S']
# })
# encoded_data = one_hot_encode(data, ['Color', 'Size'])
# print(encoded_data)

data = np.array([['Red'], ['Green'], ['Blue'], ['Green']])
encoder = OneHotEncoder(sparse_output=False)
encoded_data = encoder.fit_transform(data)
print(encoded_data)

# using pandas
df = pd.DataFrame({"Color": ["Red", "Green", "Blue"]})

encoded_df = pd.get_dummies(df)
print(encoded_df)
print(encoded_df.astype(int))
# print vocuabulary
print("Vocabulary:", encoder.categories_[0])