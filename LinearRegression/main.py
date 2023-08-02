import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from dotenv import find_dotenv,load_dotenv
from cost_function import cost_function
from hypothesis_function import hypothesis

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

pathToLengthAndPriceDS = os.getenv("pathToLengthAndPriceDS")

data = pd.read_csv(pathToLengthAndPriceDS)


theta0 = np.arange(-200,200,0.1)
theta1 = np.arange(-200,200,0.1)
j = cost_function(theta0,theta1,data)

theta1_curr = 190
current_pos = 190,cost_function(190,190,data)
learning_rate2 = 0.0001
learning_rate1 = 0.0001

for _ in range(100000):
    n = len(data)
    derv1 = 0
    derv2 = 0
    for i in range(n):
        x = data.iloc[i].length
        y = data.iloc[i].price
        derv1 += ((current_pos[0] + theta1_curr*x) - y)
        derv2 += ((current_pos[0] + theta1_curr*x) - y)*x
    new_theta0 = current_pos[0] - (1/n)*learning_rate1*derv1
    new_theta1 = theta1_curr - (1/n)*learning_rate2*derv2

    current_pos = new_theta0,cost_function(new_theta0,new_theta1,data)
    theta1_curr = new_theta1

plt.scatter(data.length,data.price,color="black")
plt.plot(list(range(10,150)),[current_pos[0]+theta1_curr*x for x in range(10,150)],color="red")
plt.show()








