import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv("C:\\Users\\raghu\\Documents\\MLfromScratch\\datasets\\LengthAndPrice.csv")


def cost_function(theta0,theta1,points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].length
        y = points.iloc[i].price
        total_error += (y - (theta0+x*theta1)) ** 2
    return total_error/2


# eq -> h(x) = theta0 + theta1*x1
# eq2 -> J(theta) = (1/2)* summation(i=1 to m)(h(x(i))-y)^2

theta0 = np.arange(-200,200,0.1)
theta1 = np.arange(-200,200,0.1)
j = cost_function(theta0,theta1,data)

theta1_curr = 190
current_pos = 190,cost_function(190,190,data)
learning_rate2 = 0.0000001
learning_rate1 = 0.00001

for _ in range(2000):
    n = len(data)
    derv1 = 0
    derv2 = 0
    for i in range(n):
        x = data.iloc[i].length
        y = data.iloc[i].price
        derv1 += ((current_pos[0] + theta1_curr*x) - y)
        derv2 += ((current_pos[0] + theta1_curr*x) - y)*x
    new_theta0 = current_pos[0] - learning_rate1*derv1
    new_theta1 = theta1_curr - learning_rate2*derv2

    current_pos = new_theta0,cost_function(new_theta0,new_theta1,data)
    theta1_curr = new_theta1

    plt.plot(theta0,j)
    plt.scatter(current_pos[0],current_pos[1],color="red")
    plt.pause(0.001)
    plt.clf()
