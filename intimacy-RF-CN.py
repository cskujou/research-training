import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

pickle_name = "pickle-concept-net-avg"
with open(pickle_name, "rb") as f:
    X_train, y_train, X_val, y_val = pickle.load(f)

rf_reg = RandomForestRegressor(max_depth=10, max_features=None)
rf_reg.fit(X_train, y_train)
y_pred = rf_reg.predict(X_val)

mse = mean_squared_error(y_pred, y_val)
r = pearsonr(y_pred, y_val)
print(
    f"mse: {mse:0.4f}\n"
    f"rmse: {mse ** 0.5:0.4f}\n"
    f"r: {r.statistic:0.4f}"
)
