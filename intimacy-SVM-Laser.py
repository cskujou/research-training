import pickle
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

pickle_name = "pickle-laser"

with open(pickle_name, "rb") as f:
    X_train, y_train, X_val, y_val = pickle.load(f)

param = {
    "epsilon": 0.03,
    "C": 1,
    "kernel": "rbf"
}
svm_reg = SVR(**param)
svm_reg.fit(X_train, y_train)
y_pred = svm_reg.predict(X_val)

mse = mean_squared_error(y_pred, y_val)
r = pearsonr(y_pred, y_val)
print(
    f"mse: {mse:0.4f}\n"
    f"rmse: {mse ** 0.5:0.4f}\n"
    f"r: {r.statistic:0.4f}"
)
