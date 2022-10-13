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
import matplotlib as mpl

def main():
    imageFile = input("Enter name of image file: ")
    templateFile = input("Enter the name of template file: ")
    image = mpl.image.imread(imageFile)
    template = mpl.image.imread(templateFile)
    image = to_grayscale(image)
    template = to_grayscale(template)

def to_grayscale(image):
    return np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])    
if __name__ == '__main__':
    main()


