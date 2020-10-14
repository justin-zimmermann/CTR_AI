import numpy as np
from sklearn.linear_model import LinearRegression

x = np.array([3587135, 2816400, -3243719, -2680321]).reshape((-1, 1))
y = np.array([1985, 1789, 292, 438])

model = LinearRegression()

model.fit(x, y)

print('intercept:', model.intercept_)
print('slope:', model.coef_)

y_pred = model.predict(x)
print('predicted response:', y_pred, sep='\n')

x = np.array([667257, -1824289, -560583, 1246025]).reshape((-1, 1))
y = np.array([886, 272, 579, 1029])

model = LinearRegression()

model.fit(x, y)

print('intercept:', model.intercept_)
print('slope:', model.coef_)

y_pred = model.predict(x)
print('predicted response:', y_pred, sep='\n')

#intercept: 1096.3871668881495
#slope: [0.00024703]

#intercept: 720.6058956589455
#slope: [0.00024687]