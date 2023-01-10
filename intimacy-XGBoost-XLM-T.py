import pickle

from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from xgboost import XGBRegressor

pickle_name = "pickle-xlmt"

with open(pickle_name, "rb") as f:
    X_train, y_train, X_val, y_val = pickle.load(f)

param = {
    'min_child_weight': 1,
    'max_depth': 4,
    'lambda': 0.5,
    'gamma': 0,
    'eta': 0.1,
    'colsample_bytree': 0.7,
    'alpha': 1
}

svm_reg = XGBRegressor(**param)
svm_reg.fit(X_train, y_train)
y_pred = svm_reg.predict(X_val)

mse = mean_squared_error(y_pred, y_val)
r = pearsonr(y_pred, y_val)
print(
    f"mse: {mse:0.4f}\n"
    f"rmse: {mse ** 0.5:0.4f}\n"
    f"r: {r.statistic:0.4f}"
)
