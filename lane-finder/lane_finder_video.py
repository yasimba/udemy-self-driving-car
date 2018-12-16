import cv2
import time
import numpy as np

def make_coordinates(image,line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1,y1,x2,y2])# make x and y start from botton


def average_slope_intercept(image,lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2),(y1,y2),1) #fit 1st degree polynomial to x and y points and return vector of coefficients describing slope and y intercepts
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:#left side of lane reduces with increase in x
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
    left_fit_average = np.average(left_fit,axis=0)#get average slope and average intercept
    right_fit_average = np.average(right_fit,axis=0)#get average slope and average intercept
    left_line = make_coordinates(image,left_fit_average)
    right_line = make_coordinates(image,right_fit_average)
    return np.array([left_line,right_line])


def canny(image):
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) #change to grayscale image
    blur = cv2.GaussianBlur(gray,(5,5),0) # reduce noise by smoothening the image
    canny = cv2.Canny(blur,50,150)
    return canny

def display_lines(image,lines):
    line_image = np.zeros_like(image)# create black image na display line on it
    if lines is not None:
        # each line is 2d array, reshape to 1d array
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)#blue line
    return line_image

#mask out everything else except what is here
#we will later AND the mask and the canny image to get a new array and thus image which in this case
#are the lanes. Make array of all ones, get original image and get only ands that are 1
def region_of_interest(image):
    height = image.shape[0]
    #get the regions first by showing the image on matplotlib and then
    #denoting which parts you would like to use
    polygons = np.array([[(200,height),(1100,height),(550,250)]])
    mask = np.zeros_like(image)#create array of 0 with same shape as image
    cv2.fillPoly(mask,polygons,255)
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image

cap = cv2.VideoCapture("./videos/test2.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5) #more research needed here #threshold, fewest intersections needed to choose line
    averaged_lines = average_slope_intercept(frame,lines)
    line_image = display_lines(frame,averaged_lines)
    combo_image = cv2.addWeighted(frame,0.8,line_image,1,1) #combine black image with color image
    cv2.imshow('result',combo_image)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


#gradient measures the change in brightness over adjacent pixels
#strong gradient is a steep change
#small gradient is a shallow change
#edges are detected by sharp change in brightness, and hence gradient. This is
# traced as a white line

# Edge detection
#1) Change to grayscale, this only has one channel ranging from 0 to 255 instead of
# a colored image which has multiple channels. This makes it faster and cheaper

#Smoothening images
# We use Gaussian filters(blur = cv2.GaussianBlur(gray,(5,5),0)) to reduce noise in the image by smoothening it for better
# edge detection(RESEARCH: How do Gaussian Blurs work)

#Use Canny method to get strongest gradients in the image to detect edges strongly
# Finds derivative(f(x,y)) to find the changes in light intensity of the image

#line detection --> Hough transform
