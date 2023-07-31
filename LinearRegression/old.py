import matplotlib.pyplot as plt
import pandas as pd
import os
from dotenv import find_dotenv,load_dotenv

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

pathToLengthAndPriceDS = os.getenv("pathToLengthAndPriceDS")

data = pd.read_csv(pathToLengthAndPriceDS)

def hypothesis_function(theta0,theta1,points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].length
        y = points.iloc[i].price
        total_error += (y - (theta0+x*theta1)) ** 2
    return total_error/2

def gradient_descent(theta0,theta1,points):
    partial_derivative1 = 0
    partial_derivative2 = 0

    n = len(points)
    for i in range(n):
        x = points.iloc[i].length
        y = points.iloc[i].price

        partial_derivative1 += (2/n)*((theta0+(x*theta1))-y)*x
        partial_derivative2 += (2/n)*((theta0+(x*theta1))-y) 

    temptheta1 = theta1 - (0.00000001*partial_derivative1)
    temptheta0 = theta0 - (0.00001*partial_derivative2)
    return temptheta0,temptheta1

theta0 = 0
theta1 = 0

epochs = 20000
for i in range(epochs):
    theta0,theta1 = gradient_descent(theta0,theta1,data)

print(theta0,theta1)

plt.scatter(data.length,data.price,color="black")
plt.xlabel("length")
plt.ylabel("price")
plt.plot(list(range(10,150)),[theta0+(x*theta1) for x in range(10,150)],color="red")
plt.show()
