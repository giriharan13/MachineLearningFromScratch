import pandas as pd

# theta for each epoch -> theta = theta - (1/m)*learning_rate*partial_derivate 
def gradient_descent(x,y,theta0,theta1,learning_rate0,learning_rate1):
    derv0 = 0.0
    derv1 = 0.0
    n = len(x)
    for xi,yi in zip(x,y):
        print(xi,yi)
        derv0+= -2*(yi-(xi*theta1+theta0))
        derv1+= -2*xi*(yi-(xi*theta1+theta0))
    theta0 = theta0 - (1/n)*learning_rate0*derv0
    theta1 = theta1 - (1/n)*learning_rate1*derv1
    return theta0,theta1




