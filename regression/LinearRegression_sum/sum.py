# This program is written in Python3
# https://qiita.com/Sa2Knight/items/221be46f2702ae453ba9
# This program use the linear Regression

from pandas import DataFrame
from sklearn import linear_model

formulas = DataFrame([
	[0, 0],
	[0, 1],
	[0, 2],
	[1, 0],
	[1, 1],
	[1, 2],
	[2, 0],
	[2, 1],
	[2, 2]
])

answers = DataFrame([0, 1, 2, 1, 2, 3, 2, 3, 4])

model = linear_model.LinearRegression()
model.fit(formulas, answers)

predicted_answer = model.predict([[10, 20]])
print(predicted_answer)
