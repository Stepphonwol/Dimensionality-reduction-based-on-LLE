from sklearn.datasets import fetch_mldata
from sklearn.neighbors import BallTree
import numpy as np
import matplotlib.pyplot as plt

class LLE:
    def __init__(self):
        print("Fetching input data...")
        raw = fetch_mldata('iris')
        # row : samples; column : features
        self.X = raw.data[0:50]
        print(self.X.shape)
        print(self.X)
        self.N = self.X.shape[0]
        self.D = self.X.shape[1]

    def find_invariance(self):
        self.K = int(input("Number of nearest neighbors: "))
        self.invariance = []
        # for the ball tree, [N, D]
        tree = BallTree(self.X)
        for i in range(self.N):
            # gram matrix
            bold_x = np.zeros((self.D, self.K))
            neighbors = np.zeros((self.D, self.K))
            unit = np.ones((self.K, 1))
            dist, ind = tree.query([self.X[i]], k=self.K + 1)
            print(i)
            print(ind)
            for j in range(self.K):
                bold_x[:,j] = self.X[i]
                neighbors[:,j] = self.X[ind[0][j+1]]
            diff = bold_x - neighbors
            gram_matrix = np.dot(np.transpose(diff), diff)
            pinv_g = np.linalg.pinv(gram_matrix)
            weights = np.dot(pinv_g, unit) / np.dot(np.dot(unit.T, pinv_g), unit)
            print(weights.shape)
            print(weights)
            self.invariance.append(weights)


    def embedding(self):
        I = np.eye(self.N, dtype=int)
        W = np.zeros((self.N, self.N))
        #self.low_repre = []
        for weight in self.invariance:
            W[0:self.K] = weight
            target = np.dot(I - W, np.transpose(I - W))
            eigenvalues, eigenvectors = np.linalg.eig(target)
            eig_seq = np.argsort(eigenvalues)
            # discard the first eigenvalue
            eig_seq_indice = eig_seq[1:3]
            new_eig_vec = eigenvectors[:, eig_seq_indice]
            print(new_eig_vec.shape)
            print(new_eig_vec)
            plt.scatter(new_eig_vec[:,0], new_eig_vec[:,1])
            #self.low_repre.append(new_eig_vec)
        plt.show()

    def analyze(self):
        #self.KNN()
        self.find_invariance()
        self.embedding()

if __name__ == "__main__":
    lle = LLE()
    lle.analyze()
