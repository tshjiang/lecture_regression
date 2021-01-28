print("Hello World")
import pandas

from sklearn import linear_model
dataset = pandas.read_csv("dataset_3_outputs.csv")
print (dataset)
target = dataset.iloc[:,0].values

print(target)

data = dataset.iloc[:,3:9].values

print(data)


machine = linear_model.LinearRegression()


machine.fit(data,target)

print(machine)

new_data = [
    [0.01, -0.2, 0.5, 1.1, 0, 0],
    [-0.5, -0.1, 0.44, 0.9, 1, 0.5]
]

new_target = machine.predict(new_data)

print(new_target)