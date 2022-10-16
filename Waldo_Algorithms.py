"""
===============================================================================
ENGR 13300 Fall 2022

Program Description
    Program makes me sad

Assignment Information
    Assignment:     Python Group Project
    Author:         Name, login@purdue.edu
    Team ID:        LC2 - 26

Contributor:    Name, login@purdue [repeat for each]
    My contributor(s) helped me:
    [ ] understand the assignment expectations without
        telling me how they will approach it.
    [ ] understand different ways to think about a solution
        without helping me plan my solution.
    [ ] think through the meaning of a specific error or
        bug present in my code without looking at my code.
    Note that if you helped somebody else with their code, you
    have to list that person as a contributor here as well.
    
ACADEMIC INTEGRITY STATEMENT
I have not used source code obtained from any other unauthorized
source, either modified or unmodified. Neither have I provided
access to my code to another. The project I am submitting
is my own original work.
===============================================================================
"""
import numpy as np
import matplotlib.pyplot as plt
import math as m
import Waldo_Algorithms as wa
import matplotlib.patches as patches

def to_grayscale(image):
    
    return np.dot(image[...,:3], [0.299, 0.587, 0.114])

def find_shape(I, T):
    H,W = I.shape
    h,w = T.shape
    
    return H, W, h, w

def SSD(I, T):
    H, W, h, w = find_shape(I, T)
    R = np.zeros([H-h, W-w])
    print("\tCalculating R Matrix")
    for i in range(H-h):
        for j in range(W-w): 
            R[i,j] = np.sum(np.multiply(I[i:(i+h), j:(j+w)] - T, I[i:(i+h), j:(j+w)] - T))
            
    return R

def CC(I, T):
    H, W, h, w = find_shape(I, T)
    
    #I' calculation
    print("\tCalculating I' Matrix")
    IPrime = np.zeros([H, W])
    for i in range(H-h):
        for j in range(W-w):
            ISum = np.sum(I[i:(i+h), j:(j+w)])
            mean = ISum / (w * h)
            IPrime[i:(i+h), j:(j+w)] = np.subtract(I[i:(i+h), j:(j+w)], mean)
    
    #T' calculation
    print("\tCalculating T' Matrix")                    
    TSum = np.sum(T)
    TMean = TSum / (w * h)
    TPrime = T - TMean
    
    #R(i,j) calculation
    print("\tCalculating R Matrix")
    R = np.zeros([H-h, W-w])
    for i in range(H-h):
        for j in range(W-w):
            productOfIPrimeAndTPrimeMatrix = np.multiply(IPrime[i:(i+h), j:(j+w)],TPrime)
            R[i,j] = np.sum(productOfIPrimeAndTPrimeMatrix)
            
    return R
                                                                   
def NCC(I, T):
    H, W, h, w = find_shape(I, T)
    
    R = np.zeros([H-h, W-w])
    print("\tCalculating R Matrix")
    for i in range(H-h):
        for j in range(W-w):
            
            #numerator calculation
            productOfIAndTMatrix = np.multiply(I[i:(i+h), j:(j+w)], T)
            numerator = np.sum(productOfIAndTMatrix)
            
            #first component in denominator calculation
            squareOfIMatrix = np.multiply(I[i:(i+h), j:(j+w)], I[i:(i+h), j:(j+w)])
            firstComponent = np.sum(squareOfIMatrix)
            
            #second component in denominator calculation
            squareOfTMatrix = np.multiply(T, T)
            secondComponent = np.sum(squareOfTMatrix)
            
            #denominator calculation
            denominator = m.sqrt(firstComponent * secondComponent)
            
            #final calculation
            R[i,j] = numerator / denominator
    
    return R

def CM(I, T):
    print("\tRunning SSD")
    R1 = wa.SSD(I, T)
    print("\tRunning CC")
    R2 = wa.CC(I, T)
    print("\tRunning Algorithm NCC")
    R3 = wa.NCC(I, T)
    
    #finding of R2 shape
    H, W = R2.shape
    
    #finding of T shape
    h, w = T.shape
    
    #mean normalization of values
    normalized1 = -1 * mean_normalization(R1) + 1
    normalized2 = mean_normalization(R2)
    normalized3 = mean_normalization(R3)
    
    #averaging of values
    print("\tAveraging Activation Matrices")
    output = np.zeros([H,W])
    for i in range(H):
        for j in range(W):
            output[i,j] = (normalized1[i,j] + normalized2[i,j] + normalized3[i,j]) / 3
            
    return output

def optimal_location(R, T, algo):
    h, w = T.shape
    
    print("\tFinding optimal R value")
    if algo == 1:
        upperLeftLocation = np.zeros([2,])
        lowerRightLocation = np.zeros([2,])
        
        upperLeftLocation = np.where(R == np.min(R))
        print(f"\tMin = {np.min(R):.2f}")
        
        lowerRightLocation[0] = upperLeftLocation[0] - h
        lowerRightLocation[1] = upperLeftLocation[1] - w
        
        center = np.zeros([2,])
        center[0] = (upperLeftLocation[0] - lowerRightLocation[0]) / 2
        center[1] = (upperLeftLocation[1] - lowerRightLocation[1]) / 2
        
        height = h
        width = w
        
        return upperLeftLocation, height, width, center
    
    else:
        upperLeftLocation = np.zeros([2,])
        lowerRightLocation = np.zeros([2,])
        
        upperLeftLocation = np.where(R == np.max(R))
        print(f"\tMax = {np.max(R):.2f}")
        
        lowerRightLocation[0] = upperLeftLocation[0] - h
        lowerRightLocation[1] = upperLeftLocation[1] - w
        
        center = np.zeros([2,])
        center[0] = (upperLeftLocation[0] - lowerRightLocation[0]) / 2
        center[1] = (upperLeftLocation[1] - lowerRightLocation[1]) / 2
        
        height = h
        width = w
        
        return upperLeftLocation, height, width, center

def plot_image(image, upperLeftLocation, height, width):
    print("\tPlotting image")
    fig, ax = plt.subplots()
    ax.imshow(image)
    rect = patches.Rectangle((upperLeftLocation[1], upperLeftLocation[0]), width, height, fill = False, edgecolor = 'lawngreen')
    ax.add_patch(rect)
    
    plt.show()
    
def mean_normalization(M):
    output = np.divide(np.subtract(M, np.min(M)), np.subtract(np.max(M), np.min(M)))
    
    return output