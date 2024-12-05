import numpy as np

class Perceptron(object):
    def __init__(self, eta = 0.5, n_iter = 150, random_state = 1):
      self.eta = eta 
      self.n_iter = n_iter
      self.random_state = random_state  

    def fit(self, X, y):
      rgen = np.random.RandomState(self.random_state)
      self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size = 1 + X.shape[1])  
      self.errores_ = []  
      print('Pesos iniciales', self.w_)

      for _ in range(self.n_iter):
          errores = 0
          for xi, etiqueta in zip(X, y): 
            actualizacion = self.eta * (etiqueta - self.predice(xi))
            self.w_[1:] += actualizacion * xi
            self.w_[0] += actualizacion
            errores += int(actualizacion != 0)
          self.errores_.append(errores)
          print('Pesos en epoch', _ , ':', self.w_)
      return self
    
    def entrada_neta(self, X):
      'Calculo de la entrada neta'
      return np.dot(X, self.w_[1:]) + self.w_[0]

    def predice(self, X): 
      'Etiqueta de clase de retorno despues del paso unitario'
      return np.where(self.entrada_neta(X) >= 0.0, 1, -1)
  
    
  
datos = [[1,1], [1,-1], [-1,1], [-1,-1]]
X = np.array(datos)
y = [-1, 1, 1, -1]




import matplotlib.pyplot as plt

plt.scatter(X[0:3, 0], X[0:3, 1], color = 'red', marker = 'o', label = 'Positivo')
plt.scatter(X[3, 0], X[3, 1], color = 'blue', marker = 'x', label = 'Negativo')

plt.xlabel('X1')
plt.ylabel('X2')
plt.legend(loc = 'upper center')

plt.show()





ppn = Perceptron(eta = 0.5, n_iter = 5)

ppn.fit(X,y)
plt.plot(range(1, len(ppn.errores_) + 1), ppn.errores_, marker = 'o')
plt.xlabel('Epochs')
plt.ylabel('Numero de actualizaciones')
plt.show()


from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predice(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')
plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend(loc='upper left')
plt.show()