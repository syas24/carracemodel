# carracemodel(This model predicts the historical data about the car and it also tells about the efficiency and accuracy of the car.)
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
df = pd.read_csv('path/to/your/yashishree.csv')
print(df.isnull().sum())
df['time'].fillna(df['time'].median(), inplace=True)
df = pd.get_dummies(df, columns=['constructor'], drop_first=True)
df['status'] = df['status'].apply(lambda x: 1 if x == 'Finished' else 0)
numerical_features = ['grid', 'laps', 'points']
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])
average_metrics = df.groupby('driver').agg({'grid': 'mean', 'points': 'mean'}).rename(columns={'grid': 'avg_grid_position', 'points': 'avg_race_points'})
df = df.merge(average_metrics, on='driver', how='left')
df[['avg_grid_position', 'avg_race_points']] = scaler.fit_transform(df[['avg_grid_position', 'avg_race_points']])
features = ['grid', 'laps', 'points', 'status', 'avg_grid_position', 'avg_race_points'] + [col for col in df.columns if col.startswith('constructor_')]
target = 'position'  # Assuming 'position' is the target variable
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f'RMSE: {rmse}')
