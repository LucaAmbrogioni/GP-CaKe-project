
import numpy as np
from copy import deepcopy
import sys
import time
sys.setrecursionlimit(20000)


def matrix_division(divider, divided, side, cholesky):
    X = np.matrix(divided)
    if cholesky is "yes":
        M = np.matrix(np.linalg.cholesky(divider))
        if side is "right":
            first_division = np.linalg.solve(M,X.T).T
            result = np.linalg.solve(M.T,first_division.T).T
        elif side is "left":
            first_division = np.linalg.solve(M,X)
            result = np.linalg.solve(M.T,first_division)
        else:
            print("The side should be either left or right")
            return
    else:
        M = np.matrix(divider)
        if side is "right":
            result = np.linalg.solve(M.T,X.T).T
        elif side is "left":
            result = np.linalg.solve(M,X)
        else:
            print("The side should be either left or right")
            return
    return result
#
def heaviside(x):
    return 0.5*(np.sign(x) + 1)
    
def step(x, step_point, side):
    if side is "right":
        s = +1
    elif side is "left":
        s = -1
    return heaviside(s*x - s*step_point)

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x)>0)

def nested_zip(*list_matrices):
    zipped_matrix = []
    for first_index,_ in enumerate(list_matrices[0]):
        rows_list = []
        for matrix in list_matrices:
            rows_list += [matrix[first_index]]
        zipped_row = zip(*rows_list)
        zipped_matrix += [zipped_row]
    return zipped_matrix  

def fill_diagonal(matrix, entry):    
    filled_matrix = [deepcopy(row) for row in matrix]    
    for index in range(0, len(filled_matrix)):
        filled_matrix[index][index] = entry
    return filled_matrix

def nested_map(function, matrix, ignore_diagonal = False):
    apply_function = lambda item,index1,index2: function(item) if ((not ignore_diagonal) or index1 != index2) else item
    return [[apply_function(item,row_index,column_index) 
             for column_index, item in enumerate(row)]
            for row_index, row in enumerate(matrix)
            ]

def nested_reduce(function, matrix):
    return reduce(function, [reduce(function,row) for row in matrix])

def foldRight(function, initial_value, lst):
    if len(lst) == 0:
        return initial_value
    else:
        return function(lst[0], foldRight(function, initial_value, lst[1:]))

def nested_foldRight(first_function, second_function, initial_value, matrix):
    return reduce(second_function, [foldRight(first_function, initial_value, row) for row in matrix])

# Courtesy of https://stackoverflow.com/questions/5849800/tic-toc-functions-analog-in-python: 
def tic():
    #Homemade version of matlab tic and toc functions    
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(round(time.time() - startTime_for_tictoc,2)) + " seconds.")
    else:
        print("Toc: start time not set")

