import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

df = pd.read_csv("insurance.csv")

df["sex"] = df["sex"].map({"male":0, "female":1})
df["smoker"] = df["smoker"].map({"no":0, "yes":1})
df["region"] = df["region"].astype("category").cat.codes

# NEW FEATURE
df["company"] = [i % 3 for i in range(len(df))]

X = df.drop("charges", axis=1)
y = df["charges"]

model = RandomForestRegressor()
model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))

print("Model updated with company feature!")
