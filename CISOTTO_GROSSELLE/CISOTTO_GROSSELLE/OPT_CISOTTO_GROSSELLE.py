#CISOTTO MARTINA
#GROSSELLE ALICE


import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



###################################################################################
# GENERAL FUNCTIONS (prob, loss, accuracy, grid search)
###################################################################################

#compute probabilities for multiclass logistic regression
def compute_prob(X, A):
    Z = np.dot(A, X)

    #for numerical stability: subtract max for every row to avoid overflow with exp()
    Z_max = np.max(Z, axis=1, keepdims=True)
    num = np.exp(Z - Z_max)

    den = np.sum(num, axis=1, keepdims=True)

    return num / den


#compute the average negative log-likelihood (Loss function)
def compute_loss(X, A, b):
    m = A.shape[0]
    Z = np.dot(A, X)

    # Term 1
    T1 = -(Z[np.arange(m), b])

    # Term 2
    Z_max = np.max(Z, axis=1, keepdims=True)
    T2 = np.log(np.sum(np.exp(Z - Z_max), axis=1)) + Z_max.flatten()

    tot_loss = np.sum(T1 + T2)

    return tot_loss / m


#evaluate model performance comparing the predicted class labels with the true labels
def compute_accuracy(X, A, b):
    Z = np.dot(A, X)
    predict = np.argmax(Z, axis=1)

    return np.mean(predict == b)


#grid search for the learning rate factor q (hyperparameter) that maximize accuracy
def grid_search_q(A_train, b_train, A_test, b_test, k_classes, algorithm, q_val):
    best_q = None
    best_acc = -1

    for q in q_val:
        # use less iters for grid search
        X, h = algorithm(A_train, b_train, k_classes, iters=1000, q=q, eps=1e-1, A_test=A_test, b_test=b_test)
        final_acc = h[-1, 2] #take only last accuracy in history

        if final_acc > best_acc:
            best_acc = final_acc
            best_q = q

    print(f"Best q: {best_q}, accuracy: {best_acc:.4f}")
    return best_q



###################################################################################
# OPTIMIZATION ALGORITHMS
###################################################################################

# GRADIENT DESCENT ALGORITHM
def gradient_descent(A, b, k_classes, iters, q, eps, A_test=None, b_test=None):
    m = A.shape[0]
    d = A.shape[1]
    X = np.zeros((d, k_classes))
    indicator = np.zeros((m, k_classes))
    indicator[np.arange(m), b] = 1

    #Lipschitz constant for multiclass logistic regression
    L_tot = np.sum(A**2) / (4 * m)
    #learning rate
    alpha = q / L_tot

    history = []
    start_t = time.time()

    for i in range(iters):
        S = compute_prob(X, A)
        #compute the full gradient
        grad_matrix = (np.dot(A.T, (S - indicator)) / m)

        #stopping condition: norm of full gradient
        grad_norm = np.linalg.norm(grad_matrix)
        if grad_norm < eps:
            print(f"GRADIENT DESCENT: Iters to converge {i}, Grad norm: {grad_norm:.4f}")
            break

        #X update
        X = X - alpha * grad_matrix

        t = time.time() - start_t
        curr_loss = compute_loss(X, A, b)
        curr_acc = 0
        if A_test is not None and b_test is not None:
            curr_acc = compute_accuracy(X, A_test, b_test)
        history.append((t, curr_loss, curr_acc))

    return X, np.array(history)


# BCGD WITH GAUSS-SOUTHWELL RULE ALGORITHM
def bcgd_gauss_southwell(A, b, k_classes, iters, q, eps, A_test=None, b_test=None):
    m = A.shape[0]
    d = A.shape[1]
    X = np.zeros((d, k_classes))
    indicator = np.zeros((m, k_classes))
    indicator[np.arange(m), b] = 1

    #local Lipschitz constants for each feature (row of X)
    L_blocks = np.sum(A**2, axis=0) / (4 * m)

    history = []
    start_t = time.time()

    for i in range(iters):
        S = compute_prob(X, A)

        #compute full gradient to apply gauss southwell rule
        grad_matrix = (np.dot(A.T, (S - indicator)) / m)
        row_norms = np.linalg.norm(grad_matrix, axis=1)

        #stopping condition: norm of full gradient
        grad_norm = np.sqrt(np.sum(row_norms**2))
        if grad_norm < eps:
            print(f"BCGD GAUSS-SOUTHWELL: Iters to converge {i}, Grad norm: {grad_norm:.6f}")
            break

        #Gauss Southwell Rule
        max_row = np.argmax(row_norms)
        #learning rate
        alpha = q / L_blocks[max_row]
        #update only the row of X with the largest gradient norm
        X[max_row, :] = X[max_row, :] - alpha * grad_matrix[max_row, :]

        t = time.time() - start_t
        curr_loss = compute_loss(X, A, b)
        curr_acc = 0
        if A_test is not None and b_test is not None:
            curr_acc = compute_accuracy(X, A_test, b_test)
        history.append((t, curr_loss, curr_acc))

    return X, np.array(history)


###################################################################################
# DATA PREPARATION (Randomly generated and real dataset)
###################################################################################

# RANDOM DATA

k = 50   
d = 1000   
m = 1000   
np.random.seed(23)
A_rand = np.random.standard_normal(size = (m, d))   #random matrix A from Normal Distribution, 1000*1000
X_rand = np.random.standard_normal(size = (d, k))   #random matrix X from Normal Distribution, 1000*50
E_rand = np.random.standard_normal(size = (m, k))   #random matrix E from Normal Distribution, 1000*50
B_rand = A_rand @ X_rand + E_rand
b_rand = np.argmax(B_rand, axis = 1)  #vector with class labels by considering the max index in every B's row


# REAL DATASET (ISOLET)

data = pd.read_csv('isolet5.data', header=None)
A_data = data.iloc[:, :-1].values
b_data = data.iloc[:, -1].values.astype(int) - 1

#info about dataset
print(f"ISOLET dataset dimension: {A_data.shape}")
print(f"Number of classes: {len(np.unique(b_data))}")
#unique_classes: labels of letters
#samples_per_class: how many obs per class
unique_classes, samples_per_class = np.unique(b_data, return_counts=True)
#min and max number of obs in all classes
min_samples = np.min(samples_per_class)
max_samples = np.max(samples_per_class)
print(f"Minimum samples in a class: {min_samples}")
print(f"Maximum samples in a class: {max_samples}")



A_train_data, A_test_data, b_train_data, b_test_data = train_test_split(A_data, b_data, test_size=0.2, random_state=42)

#dataset standardization for convergence in logistic regression
scaler = StandardScaler()
A_train_sdata = scaler.fit_transform(A_train_data)
A_test_sdata = scaler.transform(A_test_data)
n_classes = len(np.unique(b_data))

#grid search for best learning rate factor q on ISOLET
possible_q = [1, 4, 7, 9, 10]

best_q_gd_data = grid_search_q(A_train_sdata, b_train_data, A_test_sdata, b_test_data, n_classes, gradient_descent, possible_q)
best_q_gs_data = grid_search_q(A_train_sdata, b_train_data, A_test_sdata, b_test_data, n_classes, bcgd_gauss_southwell, possible_q)


###################################################################################
# EXECUTION ON REAL DATASET (ISOLET)
###################################################################################

# PARAMETERS
max_iter = 2000
eps = 1e-1


# GD
X_gd_data, h_gd_data = gradient_descent(A_train_sdata, b_train_data, n_classes, max_iter, best_q_gd_data, eps, A_test_sdata, b_test_data)
# BCGD Gauss-Southwell
X_gs_data, h_gs_data = bcgd_gauss_southwell(A_train_sdata, b_train_data, n_classes, max_iter, best_q_gs_data, eps, A_test_sdata, b_test_data)


# ACCURACY
print(f"GD accuracy: {compute_accuracy(X_gd_data, A_test_sdata, b_test_data)*100:.2f}%")
print(f"BCGD GAUSS-SOUTHWELL accuracy: {compute_accuracy(X_gs_data, A_test_sdata, b_test_data)*100:.2f}%")

# PLOT accuracy vs CPU time on ISOLET
plt.figure(figsize=(10, 6))
plt.plot(h_gd_data[:, 0], h_gd_data[:, 2], label='GRADIENT DESCENT')
plt.plot(h_gs_data[:, 0], h_gs_data[:, 2], label='BCGD GAUSS-SOUTHWELL')
plt.xlabel('CPU time (seconds)')
plt.ylabel('Test accuracy')
plt.title('Accuracy vs CPU time on ISOLET')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# PLOT loss vs CPU time on ISOLET
plt.figure(figsize=(10, 6))
plt.plot(h_gd_data[:, 0], h_gd_data[:, 1], label='GRADIENT DESCENT')
plt.plot(h_gs_data[:, 0], h_gs_data[:, 1], label='BCGD GAUSS-SOUTHWELL')
plt.xlabel('CPU time (seconds)')
plt.ylabel('Loss')
plt.title('Loss vs CPU time on ISOLET', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


###################################################################################
# EXECUTION ON RANDOM DATA
###################################################################################

#fix the same hyperparameter q for the 2 algorithms
q = 1

# GRADIENT DESCENT
X_gd_rand, h_gd_rand = gradient_descent(A_rand, b_rand, k, max_iter, q, eps)
# BCGD Gauss-Southwell
X_gs_rand, h_gs_rand = bcgd_gauss_southwell(A_rand, b_rand, k, max_iter, q, eps)


# PLOT loss vs CPU time on RANDOM DATA
plt.figure(figsize=(10, 6))
plt.plot(h_gd_rand[:, 0], h_gd_rand[:, 1], label='GRADIENT DESCENT')
plt.plot(h_gs_rand[:, 0], h_gs_rand[:, 1], label='BCGD GAUSS-SOUTHWELL')
plt.xlabel('CPU time (seconds)')
plt.ylabel('Loss')
plt.title('Loss vs CPU time on Random data')
plt.legend()
plt.grid(True, alpha=0.5)
plt.show()

# PLOT training loss vs iterations on RANDOM DATA
plt.figure(figsize=(10, 6))
plt.plot(h_gd_rand[:, 1], label='GRADIENT DESCENT')
plt.plot(h_gs_rand[:, 1], label='BCGD GAUSS-SOUTHWELL')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title(f'Loss vs Iterations on Random data')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()



#analyze difference in iterations / time / accuracy between algorithms with different learning rate on ISOLET
possible_q = [1, 10]

for q in possible_q:
    print(f"Test q = {q}")

    #Gradient Descent
    X_gd, h_gd = gradient_descent(A_train_sdata, b_train_data, n_classes, max_iter, q, eps, A_test_sdata, b_test_data)
    iter_gd = len(h_gd)
    time_gd = h_gd[-1, 0]    #last value of time
    acc_gd = compute_accuracy(X_gd, A_test_sdata, b_test_data) * 100

    print("GRADIENT DESCENT")
    print("Iters:", iter_gd)
    print("Time (seconds):", round(time_gd, 2))
    print("Accuracy:", round(acc_gd, 2))


    #BCGD Gauss-Southwell
    X_gs, h_gs = bcgd_gauss_southwell(A_train_sdata, b_train_data, n_classes, max_iter, q, eps, A_test_sdata, b_test_data)
    iter_gs = len(h_gs)
    time_gs = h_gs[-1, 0]
    acc_gs = compute_accuracy(X_gs, A_test_sdata, b_test_data) * 100

    print("BCGD GAUSS-SOUTHWELL")
    print("Iters:", iter_gs)
    print("Time (seconds):", round(time_gs, 2))
    print("Accuracy:", round(acc_gs, 2))