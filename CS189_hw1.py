import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

data_1 = np.load('data/dataset_1.npy')
data_2 = np.load('data/dataset_2.npy')




X1, Y1 = data_1[:,0], data_1[:,1]
X2, Y2 = data_2[:,0], data_2[:,1]



# a.1) plot data as scatterplot

def plot_data(X1, Y1, X2, Y2):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(X1, Y1, color = 'blue')
    plt.xlabel('X1')
    plt.ylabel('Y1')
    plt.title('Dataset 1')

    plt.subplot(1, 2, 2)
    plt.scatter(X2, Y2, color = 'red')
    plt.xlabel('X2')
    plt.ylabel('Y2')
    plt.title('Dataset 2')
    plt.show()


# a.2) calculate correlation coefficient only using basic matrix calc

def calc_correlation(X1, Y1, X2, Y2):
    # sum of X1 / number of X1
    mean_x1 = np.sum(X1) / len(X1)
    mean_y1 = np.sum(Y1) / len(Y1)
    mean_x2 = np.sum(X2) / len(X2)
    mean_y2 = np.sum(Y2) / len(Y2)

    cov_x1y1 = np.sum((X1 - mean_x1) * (Y1 - mean_y1)) / len(X1)
    cov_x2y2 = np.sum((X2 - mean_x2) * (Y2 - mean_y2)) / len(X2)

    sd_x1 = np.sqrt(np.sum((X1 - mean_x1)**2) / len(X1))
    sd_y1 = np.sqrt(np.sum((Y1 - mean_y1)**2) / len(Y1))
    sd_x2 = np.sqrt(np.sum((X2 - mean_x2)**2) / len(X2))
    sd_y2 = np.sqrt(np.sum((Y2 - mean_y2)**2) / len(Y2))

    corr_x1y1 = cov_x1y1 / (sd_x1 * sd_y1)
    corr_x2y2 = cov_x2y2 / (sd_x2 * sd_y2)

    print("Correlation coefficient x1y1:", corr_x1y1)
    print("Correlation coefficient x2y2:", corr_x2y2)

calc_correlation(X1, Y1, X2, Y2)

def get_mse(Y, y_hat):
    return np.sum((Y - y_hat)**2) / len(Y)

# c.1) implement w* = (X^T*X)^-1 * X^T * Y
def calc_w_1d(X, Y):
    # Transpose of X1 is equal to X1 (1D)
    # (X^T*X)^-1 is equal to 1 / (X*X) for 1D
    xtx_inv = 1 / (np.dot(X, X))

    # w* = (X^T*X)^-1 * X^T * Y = 1 / (X*X) * X * Y = Y / X
    w_star = np.dot(xtx_inv, np.dot(X, Y))

    # Compute the predictions y_hat = X * w*
    y_hat = np.dot(X, w_star)

    # Compute MSE
    mse = np.sum((Y - y_hat)**2) / len(Y)

    # Print results
    print("w_star:", w_star)
    print("MSE for dataset:", mse)

    return w_star


def calc_w(X, Y):
    # (X^T*X)^-1
    xtx_inv = np.linalg.inv(np.dot(X.T, X))

    # w* = (X^T*X)^-1 * X^T * Y
    w_star = np.dot(np.dot(xtx_inv, X.T), Y)

    # Compute the predictions y_hat = X * w*
    y_hat = np.dot(X, w_star)

    # Compute MSE
    mse = get_mse(Y, y_hat)

    # Print results
    print("w_star:", w_star)
    print("MSE for dataset:", mse)

    return w_star


def plot_data_and_line_1(X, Y, w_star):
    # plot line for dataset 1 with univormily spaced x e [0, 10]
    x = np.linspace(0, 10, 100)
    y = w_star * x

    plt.figure(figsize=(10, 5))
    plt.scatter(X, Y)
    plt.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('Y1')
    plt.title('Dataset 1')
    plt.show()

def plot_data_and_line_2(X, Y, w_star):
    # plot line for dataset 1 with univormily spaced x e [0, 10]
    x = np.linspace(0, 10, 100)
    y = w_star[0] * x + w_star[1]

    plt.figure(figsize=(10, 5))
    plt.scatter(X, Y)
    plt.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('Y1')
    plt.title('Dataset 1')
    plt.show()

def plot_data_and_line_3(X, Y, w_star):
    # plot line for dataset 1 with univormily spaced x e [0, 10]
    x = np.linspace(0, 10, 100)
    y = w_star[0] * x**2 + w_star[1] * x + w_star[2]

    plt.figure(figsize=(10, 5))
    plt.scatter(X, Y)
    plt.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('Y1')
    plt.title('Dataset 1')
    plt.show()




'''
# d) Transform the features with phi(x_i) = [x_i, 1]
# Add a column of ones to X1
phi1 = np.column_stack((X1, np.ones(len(X1))))

w1_star = calc_w(phi1, Y1)
plot_data_and_line_2(X1, Y1, w1_star)

# phi(x_i) = [x_i^2, x_i, 1]
phi2 = np.column_stack((X1**2, X1, np.ones(len(X1))))

w2_star = calc_w(phi2, Y1)
plot_data_and_line_3(X1, Y1, w2_star)

'''

'''
# f) repeat for dataset 2

w_star = calc_w_1d(X2, Y2)
plot_data_and_line_1(X2, Y2, w_star)

phi1 = np.column_stack((X2, np.ones(len(X2))))
w1_star = calc_w(phi1, Y2)
plot_data_and_line_2(X2, Y2, w1_star)

phi2 = np.column_stack((X2**2, X2, np.ones(len(X2))))
w2_star = calc_w(phi2, Y2)
plot_data_and_line_3(X2, Y2, w2_star)

'''



#a)
plot_data(X1, Y1, X2, Y2)

w_star_2 = calc_w_1d(X2, Y2)
plot_data_and_line_1(X2, Y2, w_star_2)


# g) generate 5 polynomial vectors from dataset 2 up until 5th degree and calculate w* for each

phi1 = np.column_stack((X2, np.ones(len(X2))))
phi2 = np.column_stack((X2**2, X2, np.ones(len(X2))))
phi3 = np.column_stack((X2**3, X2**2, X2, np.ones(len(X2))))
phi4 = np.column_stack((X2**4, X2**3, X2**2, X2, np.ones(len(X2))))
phi5 = np.column_stack((X2**5, X2**4, X2**3, X2**2, X2, np.ones(len(X2))))

# generate w*
w1_star = calc_w(phi1, Y2)
w2_star = calc_w(phi2, Y2)
w3_star = calc_w(phi3, Y2)
w4_star = calc_w(phi4, Y2)
w5_star = calc_w(phi5, Y2)

plot_data_and_line_2(X2, Y2, w1_star)

plot_data_and_line_3(X2, Y2, w2_star)


# k fold validation for k = 4 and compute average training and validation error

phi_set = [phi1, phi2, phi3, phi4, phi5]
k = 4

def k_fold_validation(X, Y, k):
    kf = KFold(k, shuffle=True, random_state = 100)
    train_errors = []
    val_errors = []

    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        Y_train, Y_val = Y[train_index], Y[val_index]

        fold_train_errors = []
        fold_val_errors = []

        for phi in phi_set:
            phi_train = phi[train_index]
            phi_val = phi[val_index]

            w_star = calc_w(phi_train, Y_train)

            y_train_pred = np.dot(phi_train, w_star)
            y_val_pred = np.dot(phi_val, w_star)

            fold_train_errors.append(get_mse(Y_train, y_train_pred))
            fold_val_errors.append(get_mse(Y_val, y_val_pred))

        train_errors.append(fold_train_errors)
        val_errors.append(fold_val_errors)           

    train_errors = np.array(train_errors)
    val_errors = np.array(val_errors)

    avg_train_errors = np.mean(train_errors, axis=0)
    avg_val_errors = np.mean(val_errors, axis=0)

    return avg_train_errors, avg_val_errors


avg_train_errors, avg_val_errors = k_fold_validation(X2, Y2, k)

print("Average training errors:", avg_train_errors)
print("Average validation errors:", avg_val_errors)

# lot the training and validation MSE as a function of the polynomial degree

plt.figure(figsize=(10, 5))
plt.plot(range(1, 6), avg_train_errors, label='Training Error')
plt.plot(range(1, 6), avg_val_errors, label='Validation Error')

plt.yscale('log')
plt.xlabel('Polynomial Degree')
plt.ylabel('MSE')
plt.title('Training and Validation Error vs Polynomial Degree')
plt.legend()
plt.show()




