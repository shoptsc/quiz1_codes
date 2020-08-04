import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, r2_score


df = pd.read_csv('DSN PROJECT/energydata.csv')

df = df.drop(columns=['date', 'lights'])
print(df.describe())
print(df.info())
print((df.columns))
print(df.corr())

scalarr = MinMaxScaler()
norm_df = pd.DataFrame(scalarr.fit_transform(df), columns = df.columns)

x = norm_df[['T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3', 'T4', 'RH_4',
        'T5', 'RH_5', 'T6', 'RH_6', 'T7', 'RH_7', 'T8', 'RH_8', 'T9', 'RH_9',
        'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed', 'Visibility',
        'Tdewpoint','rv1', 'rv2']]

y = norm_df[['Appliances']]


# Training and Testing the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

# Fitting the Multiple Regression Model
reg = LinearRegression()
reg.fit(x_train, y_train)
coeff = reg.coef_
intercept = reg.intercept_
print(np.sort(coeff))

y_pred = reg.predict(x_test)
y_train_pred = reg.predict(x_train)

Rsqrd_train = explained_variance_score(y_train, y_train_pred)

Rsqrd = explained_variance_score(y_test, y_pred)
Rsqrw = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
rmse1 = np.sqrt(mse)
rss = np.sum(np.square(y_test - y_pred))

ridge = Ridge(alpha = 0.4, normalize = True)
ridge.fit(x_train, y_train)
ridge_coeff = ridge.coef_
rideg_intercept = ridge.intercept_
ridge_pred = ridge.predict(x_test)
ridge_mse = mean_squared_error(y_test, ridge_pred)
rideg_rmse1 = np.sqrt(ridge_mse)

las = Lasso(alpha = 0.001, normalize = True)
las.fit(x_train, y_train)
lasso_coeff = las.coef_
lasso_intercept  =las.intercept_
lasso_pred = las.predict(x_test)
lasso_mse = mean_squared_error(y_test, lasso_pred)
lasso_rmse1 = np.sqrt(lasso_mse)
print(round(lasso_rmse1, 3))
X = norm_df[['T2']]
Y = norm_df[['T6']]

X_train, X_test, Y_train,Y_test = train_test_split(X, Y, test_size = 0.3, random_state=42)

lnr = LinearRegression()
lnr.fit(X_train, Y_train)
y_predd = lnr.predict(X_test)

Rsrq = explained_variance_score(Y_test, y_predd)
Rsqr = r2_score(Y_test, y_predd)
mae1 = mean_absolute_error(Y_test, y_predd)
rss1 = np.sum(np.square(Y_test - y_predd))
mse1 = mean_squared_error(Y_test, y_predd)
rmse11 = np.sqrt(mse1)



print(round(Rsrq, 2))
print(round(mser, 2))
print(round(rss1, 2))
print(round(rmse11, 2))



