import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from tabulate import tabulate
import math

class MatrixMultiplication():
    """
    Class to compute matrix multiplication using random sampling.

    Attributes:
        A (np.ndarray): Left matrix operand.
        B (np.ndarray): Right matrix operand.
        r (int): Number of columns to sample, default is 20.
        sampling (str): Type of sampling; 'uniform' for uniform sampling or None for proportional sampling based on norms.

    Methods:
        sample: Selects indices based on the specified sampling method.
        computation: Computes the estimated matrix multiplication.
        loss: Computes the relative error of the multiplication.
        act: Orchestrates the sampling, computation, and error calculation.
    """
    def __init__(self, A, B, r=20, sampling=None):
        self.A = A
        self.B = B
        self.m = A.shape[0]
        self.n = A.shape[1]
        self.p = B.shape[1]
        if A.shape[1] != B.shape[0]:
            raise ValueError("Dimension Wrong")
        self.r = r
        if sampling == "uniform":
            self.sampling = 0
        else:
            self.sampling = 1

    def sample(self):
        '''Get the distribution of columns, and then sampling.'''
        if self.sampling == 0:
            self.prob = np.ones(self.n)/self.n
        else:
            self.prob = self.p_k()
        self.indices = list(np.random.choice(list(range(self.n)), p=self.prob, replace=True, size=self.r).astype("int"))

    def computation(self):
        '''Compute estimated matrix multiplication according to selected columns.'''
        sum = np.zeros((self.m, self.p))
        for i in self.indices:
            sum = sum + self.A[:,[i]].dot(self.B[[i],:])/(self.r*self.prob[i])
        return sum

    def p_k(self):
        '''Compute p_k (when not using uniform distribution).'''
        norm_A = np.sqrt(np.sum(self.A**2, axis=0))
        norm_B = np.sqrt(np.sum(self.B**2, axis=-1))
        return norm_A*norm_B/np.sum(norm_A*norm_B)

    def loss(self):
        '''Compute the relative error. (Approximated Error)'''
        return np.linalg.norm(self.output-self.A.dot(self.B))/(np.linalg.norm(self.A)*np.linalg.norm(self.B))

    def act(self):
        """
        Get two output of this class: loss and output matrix
        """
        self.sample()
        self.output = self.computation()
        self.loss = self.loss()
        return self.loss, self.output


def problem3(k=20):
    np.random.seed(2022)
    A = pd.read_csv("./STA243_homework_1_matrix_A.csv", header=None).to_numpy()
    B = pd.read_csv("./STA243_homework_1_matrix_B.csv", header=None).to_numpy()
    result1, result2, result3, result4 = [], [], [], []
    output1, output2, output3, output4 = [], [], [], []
    for _ in range(k):
        # Do each type 20 times and take average as estimated relative error.
        # Theoretically, expectation of relative error is proportional to square root of 1/r.
        loss1, out1 = MatrixMultiplication(A,B,r=20).act()
        loss2, out2 = MatrixMultiplication(A,B,r=50).act()
        loss3, out3 = MatrixMultiplication(A,B,r=100).act()
        loss4, out4 = MatrixMultiplication(A,B,r=200).act()
        result1.append(loss1)
        result2.append(loss2)
        result3.append(loss3)
        result4.append(loss4)
        output1.append(out1)
        output2.append(out2)
        output3.append(out3)
        output4.append(out4)
    r = [20, 50, 100, 200]
    """
    This is the plot part of Problem 3 (error)
    """
    # get plot of error
    error = [np.mean(result1), np.mean(result2), np.mean(result3), np.mean(result4)]
    plt.figure(figsize=(10, 6))  # Set the dimensions of the figure for better visibility
    plt.plot(r, error, '-o', label='Relative Error', color='darkblue', markersize=8,
             linewidth=2)  # Customize line and markers
    plt.xlabel("Selected Columns", fontsize=14)  # Set the x-axis label with an increased font size
    plt.ylabel("Relative Error", fontsize=14)  # Set the y-axis label with an increased font size
    plt.title("Plot of Relative Error vs Selected Columns", fontsize=16)  # Add a title with a larger font size
    plt.grid(True)  # Enable grid for easier value estimation
    plt.legend(fontsize=12)  # Display legend with adjusted font size
    plt.xticks(r, [f"{x} Columns" for x in r])  # Customize x-axis tick labels for better description
    plt.savefig('./figures/problem3_1.png')  # Save the figure to a file
    plt.clf()  # Clear the figure canvas
    # get a dataframe to store error
    df = pd.DataFrame({
        "Selected Columns": ['r=20', 'r=50', 'r=100', 'r=200'],
        "Relative Error": error,
        "Theoretical Error": [math.sqrt(1/20),math.sqrt(1/50), math.sqrt(1/100), math.sqrt(1/200)]
    })

    # get chart using tabulate
    print(tabulate(df, headers='keys', tablefmt='psql'))

    """
       This is the plot part of Problem 3 (matrix figure)
    """
    # Create a 2x3 grid of subplots
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))  # Set a larger figure size for better visualization

    # Display the matrix multiplication result in the first subplot
    """
    Note varies of choice for cmap (way of figure RGB): grey, jet, viridis
    """
    axs[0, 0].imshow(A.dot(B), cmap='grey')  # Use a consistent color map
    axs[0, 0].set_title('Original', fontsize=14)  # Set title with a larger font size

    # Display first approximation output
    axs[0, 1].imshow(output1[3], cmap='grey')  # Consistent color map for better visual comparison
    axs[0, 1].set_title('r=20', fontsize=14)  # Set title with a larger font size

    # Display second approximation output
    axs[0, 2].imshow(output2[3], cmap='grey')
    axs[0, 2].set_title('r=50', fontsize=14)

    # Display third approximation output
    axs[1, 0].imshow(output3[3], cmap='grey')
    axs[1, 0].set_title('r=100', fontsize=14)

    # Display fourth approximation output
    axs[1, 1].imshow(output4[3], cmap='grey')
    axs[1, 1].set_title('r=200', fontsize=14)

    # Remove the unused subplot in the second row, third column
    fig.delaxes(axs[1][2])

    # Adjust the layout to make room for title and axes, avoiding overlap
    fig.tight_layout()

    # Save the figure to a file
    fig.savefig('./figures/problem3_2.png')

    # Clear the figure to free memory
    fig.clf()

def power_iteration(A, v0, eps = 1e-6, maxiter=100):
    "This is for problem 4"
    """
    Please implement the function power_iteration that takes in the matrix X and initial vector v0 and returns the eigenvector.
    A: np.array (d, d)
    v0: np.array (d,)
    """
    v1 = v0 / np.linalg.norm(v0, 2)  # Normalize initial vector
    for _ in range(maxiter):
        v0 = v1.copy()
        v1 = A.dot(v0)
        v1 /= np.linalg.norm(v1, 2)  # Normalize
        if np.linalg.norm(v1 - v0, 2) < eps:
            break
    return v1

def problem4():
    "This is given by homework python files"
    np.random.seed(2022)
    E = np.random.normal(size=(10,10))
    v = np.array([1]+[0]*9)
    lams = np.arange(1, 11)
    prods = []
    for lam in lams:
        X = lam*np.outer(v,v) + E
        v0 = np.ones(10)
        v0 = v0/np.linalg.norm(v0,2)
        vv = power_iteration(X, v0)
        prods.append(np.abs(v @ vv))

    plt.plot(lams, prods, '-ok')
    plt.xlabel('lambda')
    plt.ylabel('product')
    plt.savefig('./figures/problem4.png')
    plt.show()
    plt.clf()


class SketchedLeastSquare():
    '''
    Least Square Using Sketching Methods.
    The order is DX->HDX->S'HDX.
    '''

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.n = X.shape[0]
        self.d = X.shape[1]
        self.Transform()

    def Transform(self):
        '''Store the results of HDX and HDY.'''
        # Compute DX and DY. Since D is diagonal, we just perform matrix broadcast instead of multiplication.
        self.D = (2 * np.round(np.random.rand(self.n), 0) - 1).reshape(-1, 1)
        X_1 = self.D * self.X
        Y_1 = self.D * self.Y

        # Perform Fast Walsh–Hadamard Transform on DX/DY instead of D. Time complexity is therefore O(d*n*log(n)) otherwise it would be O(n^2*log(n)). Hence d=20, n=1048576 it is incridibale to doing this.
        # May run out time if just using D
        # Then divided by 1/sqrt(n) for normalization.
        self.fwht(X_1)
        self.fwht(Y_1)

        self.HDX = X_1 / np.sqrt(self.n)
        self.HDY = Y_1 / np.sqrt(self.n)

    def select(self, epsilon):
        '''S^T*A is equivalent to pick A's rows '''
        r = np.round(self.d * np.log(self.n) / epsilon, 0).astype("int")
        selected = np.random.choice(range(self.n), size=r, replace=True)
        X1 = np.zeros((r, self.d))
        Y1 = np.zeros((r, 1))
        for i in range(len(selected)):
            X1[i] = self.HDX[selected[i]]
            Y1[i] = self.HDY[selected[i]]
        X1 = np.sqrt(self.n / r) * X1
        Y1 = np.sqrt(self.n / r) * Y1
        return X1, Y1

    @staticmethod
    def fwht(a) -> None:
        """
        Fast Walsh–Hadamard Transform .
        """
        h = 1
        while h < len(a):
            for i in range(0, len(a), h * 2):
                for j in range(i, i + h):
                    a[j], a[j + h] = a[j] + a[j + h], a[j] - a[j + h]
            h *= 2

    @staticmethod
    def OLS(A, B):
        '''Compute time for OLS'''
        start_ols = time.time()
        np.linalg.inv(A.T.dot(A)).dot(A.T).dot(B)
        end_ols = time.time()
        return end_ols - start_ols

def problem5(n=1048576, p=20):
    np.random.seed(2022)
    X = np.random.rand(n,p)
    Y = np.random.rand(n,1)
    epsilons = [0.1, 0.05, 0.01, 0.001]
    # Calculating time.
    times = []
    SLS = SketchedLeastSquare(X,Y)
    for i in range(len(epsilons)):
        X_1, Y_1 = SLS.select(epsilons[i])
        times.append(SLS.OLS(X_1,Y_1))
    times.append(SLS.OLS(X,Y))
    method_list = [r"$\epsilon=0.1$", r"$\epsilon=0.05$", r"$\epsilon=0.01$", r"$\epsilon=0.001$", "OLS"]
    df = pd.DataFrame({"method list": method_list,
        "Running Time ": times,
    })
    # get chart using tabulate
    print(tabulate(df, headers='keys', tablefmt='psql'))


if __name__ == '__main__':
    #problem3()
    #problem4()
    problem5()



