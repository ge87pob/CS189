import numpy as np
import numpy.linalg as LA
import pickle
from PIL import Image

def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train = pickle.load(open('x_train.p', 'rb'), encoding='latin1')
    y_train = pickle.load(open('y_train.p', 'rb'), encoding='latin1')
    x_test = pickle.load(open('x_test.p', 'rb'), encoding='latin1')
    y_test = pickle.load(open('y_test.p', 'rb'), encoding='latin1')
    return x_train, y_train, x_test, y_test

def visualize_data(images: np.ndarray, controls: np.ndarray) -> None:
    """
    Args:
        images (ndarray): image input array of size (n, 30, 30, 3).
        controls (ndarray): control label array of size (n, 3).
    """
    # Current images are in float32 format with values between 0.0 and 255.0
    # Just for the purposes of visualization, convert images to uint8
    images = images.astype(np.uint8)

    # visualize 0th, 10th, and 20th images with their corresponding control vector
    for i in [0, 10, 20]:
        img = Image.fromarray(images[i])
        img.show()
        print("Control Vector for " + str(i) + "-th image:" , controls[i])

    

def compute_data_matrix(images: np.ndarray, controls: np.ndarray, standardize: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Args:
        images (ndarray): image input array of size (n, 30, 30, 3).
        controls (ndarray): control label array of size (n, 3).
        standardize (bool): boolean flag that specifies whether the images should be standardized or not

    Returns:
        X (ndarray): input array of size (n, 2700) where each row is the flattened image images[i]
        Y (ndarray): label array of size (n, 3) where row i corresponds to the control for X[i]
    """
    # flatten x to n, 2700 array
    n = images.shape[0]
    X = images.reshape(n, -1)

    return X if not standardize else X * (1/255)*2 -1, controls


def ridge_regression(X: np.ndarray, Y: np.ndarray, lmbda: float) -> np.ndarray:
    """
    Args:
        X (ndarray): input array of size (n, 2700).
        Y (ndarray): label array of size (n, 3).
        lmbda (float): ridge regression regularization term

    Returns:
        pi (ndarray): learned policy of size (2700, 3)
    """
    # Compute pi = (X^T * X + lambda * I)^-1 * X^T * Y
    D = X.shape[1]
    pi = np.dot(np.dot(LA.inv(np.dot(X.T, X) + lmbda * np.eye(D)), X.T), Y)

    return pi


def ordinary_least_squares(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Args:
        X (ndarray): input array of size (n, 2700).
        Y (ndarray): label array of size (n, 3).

    Returns:
        pi (ndarray): learned policy of size (2700, 3)
    """ 

    # Compute pi = (X^T * X)^-1 * X^T * Y
    pi = np.dot(np.dot(LA.inv(np.dot(X.T, X)), X.T), Y)

    return pi

    

def measure_error(X: np.ndarray, Y: np.ndarray, pi: np.ndarray) -> float:
    """
    Args:
        X (ndarray): input array of size (n, 2700).
        Y (ndarray): label array of size (n, 3).
        pi (ndarray): learned policy of size (2700, 3)

    Returns:
        error (float): the mean Euclidean distance error across all n samples
    """
    # Compute average squared Euclidian distance

    n = X.shape[0]
    avg_Euclidian = 0
    for i in range(n):
        avg_Euclidian += LA.norm(np.dot(X[i], pi) - Y[i])**2
    avg_Euclidian /= n

    return avg_Euclidian

   

    


def compute_condition_number(X: np.ndarray, lmbda: float) -> float:
    """
    Args:
        X (ndarray): input array of size (n, 2700).
        lmbda (float): ridge regression regularization term

    Returns:
        kappa (float): condition number of the input array with the given lambda
    """
    # Compute condition number of X^T * X + lambda * I


    D = X.shape[1]

    condition_number = LA.cond(np.dot(X.T, X) + lmbda * np.eye(D))

    return condition_number

if __name__ == '__main__':

    x_train, y_train, x_test, y_test = load_data()
    print("successfully loaded the training and testing data")

    X, Y = compute_data_matrix(x_train, y_train)

    LAMBDA = [0.1, 1.0, 10.0, 100.0, 1000.0]


    # visualize 0th, 10th, and 20th images with their corresponding controls

    # # a) Visualize data
    print("\nexcercise a")
    visualize_data(x_train, y_train)

    # b) OLS
    print("\nexcercise b")
    X, Y = compute_data_matrix(x_train, y_train)

    try:
        # Attempt to compute the OLS solution
        pi_ols = ordinary_least_squares(X, Y)
        print("OLS solution: ", pi_ols)

    except LA.LinAlgError as e:
        # Handle the case where the matrix is singular or not invertible
        print(f"OLS failed due to a matrix inversion error: {e}")

    # c) Ridge regression
    print("\nexcercise c")
    X, Y = compute_data_matrix(x_train, y_train)
    for i in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        pi = ridge_regression(X, Y, i)
        error = measure_error(X, Y, pi)
        print("Ridge regression with lambda = ", i, " has error = ", error) 
   
    

    # d) standardize and repeat c)
    print("\nexcercise d")
    X, Y = compute_data_matrix(x_train, y_train, standardize=True)
    for i in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        pi = ridge_regression(X, Y, i)
        error = measure_error(X, Y, pi)
        print("Standardized ridge regression with lambda = ", i, " has error = ", error)  
     

    # e) evaluate with and without standardization on test data
    print("\nexcercise e")
    X, Y = compute_data_matrix(x_train, y_train)
    X_test, Y_test = compute_data_matrix(x_test, y_test)
    for i in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        pi = ridge_regression(X, Y, i)
        error = measure_error(X_test, Y_test, pi)
        print("Ridge regression (without standardization), lambda = ", i, " has error = ", error)
    
    X, Y = compute_data_matrix(x_train, y_train, standardize=True)
    X_test, Y_test = compute_data_matrix(x_test, y_test, standardize=True)
    for i in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        pi = ridge_regression(X, Y, i)
        error = measure_error(X_test, Y_test, pi)
        print("Ridge regression (with standardization), lambda = ", i, " has error = ", error)

     

    # f) condition number for lmbda = 100 with and without standardization
    print("\nexcercise f")
    X, Y = compute_data_matrix(x_train, y_train)
    print("Condition number for lambda = 100: ", compute_condition_number(X, 100))

    X, Y = compute_data_matrix(x_train, y_train, standardize=True)
    print("Condition number for lambda = 100 (standardized): ", compute_condition_number(X, 100))

    