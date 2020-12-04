import pickle
import numpy as np

final = pickle.load(open("model.p", "rb"))
data = np.genfromtxt('data.csv', delimiter=',')
y_hat = final.predict(data)
print('predictions:', y_hat)
