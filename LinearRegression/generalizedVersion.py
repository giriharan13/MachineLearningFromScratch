import pandas as pd
import numpy as np
import os
from dotenv import find_dotenv,load_dotenv
import matplotlib.pyplot as plt

dotenv_file = find_dotenv()
load_dotenv(dotenv_file)

pathtoDS = os.getenv("pathToLengthAndPriceDS")

data = pd.read_csv(pathtoDS)

def find_all_features_except_target(data,row):
    x = [1] # X0 = 1
    for i in range(len(data.columns)-1):
        x.append(data.iloc[row,i])
    return x

def find_hypothesis(thetas,x_values):
    hypothesis = 0
    for i in range(len(x_values)):
        hypothesis+= thetas*x_values
    return hypothesis

def cost_function(thetas,data):
    total_error = 0
    for i in range(len(data)):
        x = find_all_features_except_target(data,i)
        y = data.iloc[i].price    # feature that you want to predict
        hypothesis = 0
        for j in range(len(x)):
            hypothesis+= x[j]*thetas[j]
        total_error+= (hypothesis-y)**2
    return total_error/2
        
def gradientDescent(thetas,learning_factors,data):
    partial_derivatives = []
    for i in range(len(data.columns)):
        #x = find_all_features_except_target(data,i)
        #y = data.iloc[i].price
        pd = 0
        for ind in range(len(data)):
            x = find_all_features_except_target(data,ind)
            y = data.iloc[ind].price
            hypothesis = 0
            for j in range(len(x)):
                hypothesis += x[j]*thetas[j]
                pd += (hypothesis - y)*x[j]
        partial_derivatives.append(pd)
    for i in range(len(thetas)):
        thetas[i] = thetas[i] - learning_factors[i]*partial_derivatives[i]
    #print(thetas)
    return thetas


thetas = [0,0]
learning_factors = [0.00000001,0.00001]
epochs = 100
for i in range(epochs):
    thetas = gradientDescent(thetas,learning_factors,data)

print(thetas)

plt.scatter(data.length,data.price,color="black")
plt.plot(list(range(10,150)),[thetas[0]+(x*thetas[1]) for x in range(10,150)],color="red")
plt.show()

