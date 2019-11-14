import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
X = np.linspace(0.1,1,10)
Y = X * np.log(X)
plt.plot(X,Y)
plt.show()
