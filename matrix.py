import numpy as np

def matrix_addition(A,B):
    return np.add(A,B)

def matrix_multiplication(A,B):
    return A @ B

def matrix_transpose(A):
    return np.transpose(A)

def matrix_determinant(A):
    return np.linalg.det(A)

def matrix_inverse(A):
    return np.linalg.inv(A)

if __name__ == '__main__':
    
    A = np.array([1, 2, 3, 4])
    B = np.array([5, 6, 7, 8])
    print(matrix_addition(A, B))
    
    C = np.array([8, 7, 3, 4])
    D = np.array([4, 7, 12, 0])
    print(matrix_multiplication(C, D))
    
    E = np.array([4, 8, 4, 1])
    F = np.array([[1,2],[3,4]])
    print(matrix_transpose(E))
    print(matrix_inverse(F))
    print(matrix_determinant(F))
    