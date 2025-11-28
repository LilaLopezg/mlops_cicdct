import pickle

with open("models/consumo_model_latest.pkl", "rb") as f:
    data = pickle.load(f)

print("Total features:", len(data["feature_names"]))
print("\nFeature Names:")
for i, f_name in enumerate(data["feature_names"], 1):
    print(i, f_name)
