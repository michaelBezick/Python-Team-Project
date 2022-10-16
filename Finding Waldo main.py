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
import matplotlib as mpl
import time
import Waldo_Algorithms as wa

def main():
    #input of files
    imageFile = input("Enter name of image file: ")
    templateFile = input("Enter the name of template file: ")
    
    #conversion of image to np matrix
    initialImage = mpl.image.imread(imageFile)
    initialTemplate = mpl.image.imread(templateFile)
    
    #datatype of images
    imageDType = initialImage.dtype
    templateDType = initialTemplate.dtype
    
    #conversion to uint8
    if imageDType != 'uint8':
        np.uint8(initialImage)
    if templateDType != 'uint8':
        np.uint8(initialTemplate)
        
    #conversion of images to grayscale    
    image = wa.to_grayscale(initialImage)
    template = wa.to_grayscale(initialTemplate)
    
    #user input of desired template algorithm
    print("Which template matching algorithm would you like to use?")
    desiredAlgorithm = int(input("""Enter "1" for SSD, "2" for CC, "3" for NCC, or "4" for Combined Method: """))
    
    #selection of algorithm and location of most accurate dimensions
    if desiredAlgorithm == 1:
        st = time.time()
        output = wa.SSD(image, template)
        upperLeftLocation, height, width, center = wa.optimal_location(output, template, 1)
        et = time.time()
   
    if desiredAlgorithm == 2:
        st = time.time()
        output = wa.CC(image, template)
        upperLeftLocation, height, width, center = wa.optimal_location(output, template, 2)
        et = time.time()
    
    if desiredAlgorithm == 3:
        st = time.time()
        output = wa.NCC(image, template)
        upperLeftLocation, height, width, center = wa.optimal_location(output, template, 3)
        et = time.time()

    if desiredAlgorithm == 4:
        st = time.time()
        output = wa.CM(image, template)
        upperLeftLocation, height, width, center = wa.optimal_location(output, template, 4)
        et = time.time()
        
    #plotting of bounding box
    wa.plot_image(initialImage, upperLeftLocation, height, width)
    
    #center
    print(f"\nImage located at {center}")
    
    #time elapsed
    print(f"Algorithm took {et - st:.2f} seconds to finish")
if __name__ == '__main__':
    main()


