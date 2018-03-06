import numpy as np

M = 1000
XN = 2

# Siendo a e y (1xm) => (1xm)
def cost_function(a, y):
  return - (y*np.log(a) + (1-y) * np.log(1-a))


def sigmoid(z):
  return 1 / (1 + np.exp(-z))

def logistic_regression(X, y, learning_rate=0.01):
  W = np.zeros((XN, 1))
  b = np.zeros((1, M))

  J = np.zeros((1, M))
  for i in range(1000):
    Z = np.dot(W.T, X)  + b
    a = sigmoid(Z)
    J = cost_function(a, y)
    
    
    dW = np.dot(X, (a-y).T) / M
    db = np.sum(a - y) / M
    
    W -= learning_rate * dW
    b -= learning_rate * db
    res = np.ones((M, 1)) / M
    print("Costo promedio: {}".format(np.dot(J, res)))
  return (J, W, b)



def main():
  X = np.random.randn(XN, M)
  y = np.random.randn(1, M)
  J, W, b = logistic_regression(X , y, learning_rate=0.01)

if __name__ == '__main__':
  main()