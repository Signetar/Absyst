import math
import numpy as np
import json
            


class Toolkit():
    """
    A collection of tools for machine learning.
    """
    def min_rmse(self, dataset1, dataset2):
        # sorry for messy code...
        dataset1, dataset2 = np.array(dataset1), np.array(dataset2)
        n = len(dataset1)
        min_rmse = float('inf')
        for i in range(n):
            shifted_dataset2 = np.roll(dataset2, i)
            rmse = np.sqrt(np.mean((dataset1 - shifted_dataset2) ** 2))
            if rmse < min_rmse:
                min_rmse = rmse
        return min_rmse

    def loss(self, a: list, b: list, lf) -> float:
        """
        Using either RSME or MAE, it calculates the difference between a and b.
        """
        if type(a) != list:
            raise Exception("The input is not a list.")
        if type(a[0]) != int:
            raise Exception("The input is not a list of integers.")
        # exception
        if len(a) != len(b):
            raise Exception(f"?? hello could you please make sure that your input lists are the same size???????\nList 1: {len(a)} items\nList 2: {len(b)} items")


        n = len(a)
        if lf.lower() == 'rmse':
            summed = 0
            for i in range(0, n-1):
                summed += (a[i]-b[i])**2
            return math.sqrt(summed/n)
        
        elif lf.lower() == 'mrmse':
            return self.min_rmse(a, b)
        
        #mse
        elif lf.lower() == 'mae':
            summed = 0
            for i in range(0, n-1):
                summed += abs(a[i]-b[i])
            return summed/n

        # cross entropy loss
        elif lf.lower() == 'cel':
            predicted_true_probs = [a[i][b[i]] for i in range(len(b))]
            # Calculate the cross entropy loss
            loss = -1 * sum([math.log(p) for p in predicted_true_probs]) / len(b)
            return loss

        elif lf.lower() == 'hinge loss':
            loss = 0
            for i in range(len(a)):
                loss += max(0, 1 - a[i] * b[i])
            return loss / len(a)
        else:
            raise Exception("The module you chose does not exist. Maybe you misspelled it?")
        
    
    def distance(self, a: list, b: list, df) -> float:
        """
        Using distance metrics, it calculates the distance between a and b. (them being vectors)
        """
        if type(a) != list:
            print(a)
            raise Exception("The input is not a list.")
        if type(a[0]) != float:
            # turn it into a list of floats
            a = [float(i) for i in a]
        # exception
        if len(a) != len(b):
            raise Exception(f"?? hello could you please make sure that your input lists are the same size???????\nList 1: {len(a)} items\nList 2: {len(b)} items")

        n = len(a)
        if df.lower() == 'euclidean':
            summed = 0
            for i in range(0, n-1):
                summed += (a[i]-b[i])**2
            return math.sqrt(summed)
        
        elif df.lower() == 'minkowski':
            summed = 0
            for i in range(0, n-1):
                summed += (a[i]-b[i])**3
            return summed**(1/3)
        
        elif df.lower() == 'chebyshev':
            summed = 0
            for i in range(0, n-1):
                summed += abs(a[i]-b[i])
            return max(summed)
        
        elif df.lower() == 'cosine':
            summed = 0
            for i in range(0, n-1):
                summed += (a[i]*b[i])/(math.sqrt(a[i]**2)*math.sqrt(b[i]**2))
            return summed
        
        elif df.lower() == 'jaccard':
            summed = 0
            for i in range(0, n-1):
                summed += (a[i]*b[i])/(a[i]+b[i]-a[i]*b[i])
            return summed
        
        elif df.lower() == 'manhattan':
            summed = 0
            for i in range(0, n-1):
                summed += abs(a[i]-b[i])
            return summed

        else:
            raise Exception("The module you chose does not exist. Maybe you misspelled it?")

    def fuse(self, arr: list) -> list:
        """
        Takes in an array 'arr' of arrays, and fuses all of them into one by finding average.
        """
        fused = []
        for x in range(len(arr[0])):
            temporary = []
            for y in range(len(arr)):
                temporary.append(arr[y][x])
            fused.append(temporary)
        combined = [self.average(x) for x in fused]
        return combined


    def difference_matrix(self, arr: list, d) -> list:
        """
        Returns a 1D list of differences between consecutive elements in arr.
        """
        # update: make it so that this process happens 'depth' times.
        # make sure that the types are correct
        if type(arr) != list:
            raise Exception("The input is not a list.")
        if type(arr[0]) != int:
            raise Exception("The input is not a list of integers.")
        if d == 0:
            return arr
        matrix = []
        for x in range(len(arr)):
            if x < len(arr) - 1:
                matrix.append(arr[x+1]-arr[x])
        if (d == 1):
            return matrix
        else:
            return self.difference_matrix(arr, d - 1)

# similarity elimination variation
class AbstractClassifier(Toolkit):
    def __init__(self, abstractionDepth=1):
        self.lib = {} 
        self.rawlib = {}
        self.abstractionDepth = abstractionDepth
        self.lf = 'rmse'


    def average(self, a: list) -> int:
        """
        Returns an average of a 1D list of INT.
        """
        return sum(a)/len(a)

    
    # user-controlled functions
    def show(self, cin = "encoded_data"):
        """
        Allows user to interact with the cache of the system.
        """
        if cin == 'encoded_data':
            return self.lib

    def fit(self, X: list, y: list):
        """
        X: list of lists of integers (2D)
        y: list of integers (1D)

        Encodes training data (x, y) to self.lib dictionary suited for different classifiers.
        """
        templib = {}
        for i in range(len(X)):
            templib[str(y[i])] = []
        for i in range(len(X)):
            # collecting all X values corresponding to y.
            templib[str(y[i])].append(self.difference_matrix(X[i], self.abstractionDepth))

        self.rawlib = templib
        
        # now encode the saved data into different uses. templib to lib processing
        for key in templib.keys():
            item = templib[key]
            self.lib[key] = self.fuse(item[0:int(len(item))])


    def predict(self, X, d=-1, df = 'euclidean'):
        self.df = df
        if d == -1:
            d = self.abstractionDepth
        # list of predictions for X.
        out = []
        for loop in range(len(X)):
            scores = {}
            for key in self.lib.keys():
                a = self.lib[key]
                scores[key] = self.distance(a, self.difference_matrix(X[loop], d), self.df)

            out.append(int(sorted(scores, key=scores.get)[0]))
        
        return out


    def save(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.lib, f)

    def load(self, filename):
        with open(filename, 'r') as f:
            self.lib = json.load(f)