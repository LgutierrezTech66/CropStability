##Crop stability predictability model
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('crop_data.csv')
X = data.drop('stability_percentage', axis=1)
y = data['stability_percentage']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f'MAE: {mean_absolute_error(y_test, y_pred):.2f}')
print(f'R2 Score: {r2_score(y_test, y_pred):.2f}')

import joblib
joblib.dump(model, 'rf_crop_model.pkl')

