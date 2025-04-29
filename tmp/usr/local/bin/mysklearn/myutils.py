##############################################
# Programmer: Ian Myers
# Class: CPSC 322-02, Fall 2024
# Programming Assignment #7
# 11/11/24
# Description: This program assists functions in myclassifiers.py
##############################################

import numpy as np
import operator

def compute_euclidean_distance(v1,v2):
    """
        returns the euclidean distance of two points
    """
    return np.sqrt(sum([(v1[i] - v2[i]) ** 2 for i in range(len(v1))])) 

def get_distances(X_train, unseen_instance):
    """
        finds the distance between two points, then puts a tuple (index, distance) in 
            an array. It then sorts the array by distance
        returns the array of tuples
    """
    row_indexes_dists = []
    for i,row in enumerate(X_train):
        dist = compute_euclidean_distance(row, unseen_instance)
        row_indexes_dists.append((i,dist))
    row_indexes_dists.sort(key=operator.itemgetter(-1))
    return row_indexes_dists

def get_distances_categorical(X_train, unseen_instance):
    """
        finds the distance between two instances, then puts a tuple (index, distance) in 
            an array. It then sorts the array by distance
        returns the array of tuples
    """
    row_indexes_dists = []
    for i,row in enumerate(X_train):
        count = 0
        for j,val in enumerate(row):
            if val == unseen_instance[j]:
                count += 1
        dist = 1 - count / len(unseen_instance)
        row_indexes_dists.append((i,dist))
    row_indexes_dists.sort(key=operator.itemgetter(-1))
    return row_indexes_dists

def discretization_by_DOE(item):
    """
        recieves a table with continuous data and
        returns a table with that data separated into the categories
            of the given list
        
        args:
            table is a MyPyTable
    """
    if item <= 13:
        return 1
    elif item < 15:
        return 2
    elif item < 17:
        return 3
    elif item < 20:
        return 4
    elif item < 24:
        return 5
    elif item < 27:
        return 6
    elif item < 31:
        return 7
    elif item < 37:
        return 8
    elif item < 45:
        return 9
    else:
        return 10
    
def normalization(list, maxes, mins):
    """
    Takes the max and min of list then normalizes each value in list based on that

    args:
        list (list of list of values): is an list of list with each inner list having values to be 
            normalized based on the same indiced max and min
        maxes (list of max values) max for each column in inner list
        mins (list of min values) min for each column in inner list

    """
    normalized_list = []
    for row in list:
        normal = []
        for j,col in enumerate(row):
            item = (col - mins[j]) / (maxes[j] - mins[j])
            normal.append(item)
        normalized_list.append(normal)
    return normalized_list

def get_accuracy(y_pred, y_actual):
    """
    Returns the amount of times y_pred and y_actual match out of total
    y_pred(list of predicted vals)
    y_actual(list of actual vals)
    """
    # calculate accuracy
    correct = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_actual[i]:
            correct += 1
    acc = correct / len(y_pred)
    return acc
    
def randomize_in_place(alist, parallel_list=None):
    for i in range(len(alist)):
        # generate a random index to swap this value at i with
        rand_index = np.random.randint(0, len(alist)) # rand int in [0, len(alist))
        # do the swap
        alist[i], alist[rand_index] = alist[rand_index], alist[i]
        if parallel_list is not None:
            parallel_list[i], parallel_list[rand_index] = parallel_list[rand_index], parallel_list[i]