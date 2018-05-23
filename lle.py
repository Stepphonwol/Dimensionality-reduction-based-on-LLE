from sklearn.datasets import fetch_mldata
from sklearn.neighbors import BallTree
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import os

class LLE:
    def __init__(self):
        print("Fetching input data...")
        path = "/Users/lumindoec/Downloads/Asian"
        self.N = 0
        self.D = 0
        src_list = []
        for file in os.listdir(path):
            if not os.path.isdir(file) and file.endswith('jpg'):
                im = Image.open(path + "/" + file)
                print(file)
                src = np.array(im.convert("L"))
                self.D = np.size(src)
                src = src.flatten() / 255
                print(src.shape)
                self.N = self.N + 1
                src_list.append(src)
        self.X = np.zeros((self.N, self.D))
        for i, src in enumerate(src_list):
            self.X[i] = src
        print(self.X.shape)
        print(self.X)
        '''
        raw = fetch_mldata('iris')
        # row : samples; column : features
        self.X = raw.data[0:50]
        print(self.X.shape)
        print(self.X)
        self.N = self.X.shape[0]
        self.D = self.X.shape[1]
        '''

    def find_invariance(self):
        self.K = int(input("Number of nearest neighbors: "))
        # sparse matrix for embedding
        self.W = np.zeros((self.N, self.N))
        # for the ball tree, [N, D]
        tree = BallTree(self.X)
        for i in range(self.N):
            # gram matrix
            bold_x = np.zeros((self.D, self.K))
            neighbors = np.zeros((self.D, self.K))
            unit = np.ones((self.K, 1))
            dist, ind = tree.query([self.X[i]], k=self.K + 1)
            print(i)
            #print(ind)
            for j in range(self.K):
                bold_x[:,j] = self.X[i]
                neighbors[:,j] = self.X[ind[0][j+1]]
            diff = bold_x - neighbors
            gram_matrix = np.dot(np.transpose(diff), diff)
            pinv_g = np.linalg.pinv(gram_matrix)
            weights = np.dot(pinv_g, unit) / np.dot(np.dot(unit.T, pinv_g), unit)
            print(weights.shape)
            print(weights)
            # one to one
            for j in range(self.K):
                self.W[i][ind[0][j+1]] = weights[j]
        print(self.W[:,1])


    def embedding(self):
        I = np.eye(self.N, dtype=int)
        target = np.dot(I - self.W, np.transpose(I - self.W))
        eigenvalues, eigenvectors = np.linalg.eig(target)
        eig_seq = np.argsort(eigenvalues)
        #discard the first eigenvalue
        eig_seq_indice_2d = eig_seq[1:3]
        eig_seq_indice_3d = eig_seq[1:4]
        new_eig_vec_2d = eigenvectors[eig_seq_indice_2d]
        new_eig_vec_3d = eigenvectors[eig_seq_indice_3d]
        plt.figure(1)
        plt.subplot(211)
        #print(new_eig_vec.shape)
        #print(new_eig_vec)
        plt.scatter(new_eig_vec_2d[0].real, new_eig_vec_2d[1].real, edgecolors='white', alpha=0.5, c='b')
        plt.title("d=2")
        ax = plt.figure().add_subplot(212, projection='3d')
        ax.scatter(new_eig_vec_3d[0].real, new_eig_vec_3d[1].real, new_eig_vec_3d[2].real)
        plt.title("d=3")
        plt.show()


    def analyze(self):
        self.find_invariance()
        self.embedding()

if __name__ == "__main__":
    lle = LLE()
    lle.analyze()
