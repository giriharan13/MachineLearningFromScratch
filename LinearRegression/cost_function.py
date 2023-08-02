import pandas as pd
# cost function -> J(theta) = (1/2)* summation(i=1 to m)(h(x(i))-y)^2
def cost_function(theta0,theta1,points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].length
        y = points.iloc[i].price
        total_error += (y - (theta0+x*theta1)) ** 2
    return total_error/2