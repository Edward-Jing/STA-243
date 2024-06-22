import idx2numpy
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from tqdm import trange
from sklearn.mixture import GaussianMixture

# Load the data and choose only 0-4.
images_ori = idx2numpy.convert_from_file('train-images.idx3-ubyte')
labels_ori = idx2numpy.convert_from_file('train-labels.idx1-ubyte')
images = images_ori[labels_ori < 5]
labels = labels_ori[labels_ori < 5]

# Perform average pooling
images_simple = np.zeros((images.shape[0],14,14))
for i in range(14):
    for j in range(14):
        images_simple[:,i,j] = (images[:,2 * i,2 * j] + images[:,2 * i + 1,2 * j] + images[:,2 * i,2 * j + 1] + images[
                                                                                                                :,
                                                                                                                2 * i + 1,
                                                                                                                2 * j + 1]) / 4


def get_image_sample():
    # Plot a sample to check if we load the data correctly.
    plt.imshow(images_simple[0],cmap='gray')
    plt.savefig('fig1.png',bbox_inches='tight')  # Save the image
    plt.show()


# get_image_sample()

X = images_simple.reshape(images_simple.shape[0],-1)


def softmax(X,axis=None):
    max = np.max(X,axis=axis,keepdims=True)
    Y = np.exp(X - max)
    Y = Y / np.sum(Y,axis=axis,keepdims=True)
    return Y


def logsumexp(X,axis=None):
    # this is the log sum trick function when calculating log sum to avoid underflow
    max = np.max(X,axis=axis,keepdims=True)
    Y = np.exp(X - max)
    sum = np.sum(Y,axis=axis,keepdims=True)
    return np.log(sum) + max


def EM_update(X,m,type,max_iter=100,correction=0.05):
    # This correction rate is given in our hw3.pdf which equals to 0.05
    # always remember that we use 0.05Id + cov matrix
    n,d = X.shape
    mu = np.ones((m,1)) * np.mean(X,axis=0,keepdims=True) + np.random.randn(m,d)
    F = np.ones((n,m)) / m
    pi = np.ones(m) / m
    Normal = np.zeros((n,m))
    if type == 1:
        sigma = np.abs(np.ones((m,1)) * (np.var(X,axis=0,keepdims=True) * (1 + np.random.randn(1,d)))) + correction
    else:
        sigma = np.abs(np.ones((m,1)) * (np.var(X,axis=0,keepdims=True) * (1 + np.random.randn(1,d)))) + correction
    log_like0 = -1e10
    with trange(max_iter) as pbar:
        for _ in pbar:
            # E-step:
            # Compute normal pdf*pi
            # First take log to avoid numeric overflow
            constant = np.log(pi) - np.sum(np.log(sigma) / 2,axis=1) - d / 2 * np.log(2 * math.pi)
            T = np.sum(-(X.reshape(n,d,1) - mu.T.reshape(1,d,m)) ** 2 / (sigma.T.reshape(1,d,m) * 2),axis=1)
            Normal = T + constant
            # Update F
            F = softmax(Normal,axis=1)
            F = F / np.sum(F,axis=1,keepdims=True)
            # Compute Log-Likelihood
            log_like = np.sum(logsumexp(Normal,axis=1))
            pbar.set_description("Log-Likelihood={},pi={}".format(np.round(log_like,5),np.round(pi,3)))
            if abs(log_like - log_like0) / abs(log_like0) <= 1e-4:
                break
            log_like0 = log_like

            # M-step
            # Update pi
            pi = np.sum(F,axis=0,keepdims=False) / n
            # Update mu
            for j in range(m):
                mu[j] = np.average(X,axis=0,weights=F[:,j])
            # Update sigma
            # 1 for diagonal, 2 for spherical
            if type == 1:
                for j in range(m):
                    for s in range(d):
                        sigma[j,s] = np.average((X[:,s] - mu[j,s]) ** 2,weights=F[:,j])
                sigma = sigma + correction
            else:
                for j in range(m):
                    sigma[j,:] = np.sum(F[:,[j]] * (X - mu[[j],:]) ** 2) / (np.sum(F[:,j]) * d)
                sigma = sigma + correction
    return F


correction = 0.05
m = 5

# List of seeds to use
seeds = [6657,7470,9999]
types = [1,2]

results = []

for seed in seeds:
    for t in types:
        np.random.seed(seed)
        F1 = EM_update(X,m=m,type=t,correction=correction)
        pred_d = np.argmax(F1,axis=1)

        # Generate the confusion matrix
        confusion_matrix = pd.crosstab(pred_d,labels,rownames=['cluster'],colnames=['label'])

        # Calculate the prediction error rate
        total_samples = confusion_matrix.values.sum()
        correct_predictions = confusion_matrix.values.diagonal().sum()
        error_rate = 1 - correct_predictions / total_samples

        results.append({
            'seed': seed,
            'type': t,
            'confusion_matrix': confusion_matrix,
            'error_rate': error_rate
        })

# Display the results
for result in results:
    print(f"Seed: {result['seed']}, Type: {result['type']}")
    print("Confusion Matrix:")
    print(result['confusion_matrix'])
    print(f"Prediction Error Rate: {result['error_rate']:.4f}\n")

