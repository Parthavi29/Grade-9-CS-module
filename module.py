#Reading images and videos
def read_image():
    import cv2 as cv
    img=cv.imread('Photos/cat.jpg')
    cv.imshow('Photo',img)
    cv.waitKey(0)

def read_video():
    import cv2 as cv
    capture=cv.VideoCapture('Videos/dog.mp4')
    while True:
        isTrue,frame=capture.read()
        cv.imshow('Video',frame)
        if cv.waitKey(20)&0xFF==('d'):
          break
    capture.release()
    cv.destroyAllWindows()

#Resizing and Rescaling
def resize_image():
    import cv2 as cv
    img=cv.imread('Photos/cat.jpg')
    cv.imshow('Cat',img)
    def rescaleFrame(frame,scale=0.75):
        width=int(frame.shape[1]*scale)
        height=int(frame.shape[0]*scale)
        dimensions=(width,height)
        return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)
    resized_image=rescaleFrame(img)
    cv.imshow('Resized Cat',resized_image)
    cv.waitKey(0)

def rescale_video():
    import cv2 as cv
    capture=cv.VideoCapture('Videos/dog.mp4')
    def rescaleFrame(frame,scale=0.75):
        width=int(frame.shape[1]*scale)
        height=int(frame.shape[0]*scale)
        dimensions=(width,height)
        return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)

    while True:
       isTrue,frame=capture.read()
       cv.imshow('Dog',frame)
       frame_resized=rescaleFrame(frame)
       cv.imshow('Resized Dog',frame_resized)
       if cv.waitKey(20)&0xFF==('d'):
          break
    capture.release()
    cv.destroyAllWindows()

#Drawing
def blank():
    import cv2 as cv
    import numpy as np
    blank=np.zeros((500,500,3),dtype='uint8')
    cv.imshow('Blank',blank)
    cv.waitKey(0)

def paint():
    import cv2 as cv
    import numpy as np
    blank=np.zeros((500,500,3),dtype='uint8')
    blank[200:300,300:400]=255,0,0
    cv.imshow('Blue',blank)
    cv.waitKey(0)

def rect():
    import cv2 as cv
    import numpy as np
    blank=np.zeros((500,500,3),dtype='uint8')
    cv.rectangle(blank,(0,0),(blank.shape[1]//2,blank.shape[0]//2),(255,0,0),thickness=3)
    cv.imshow('Rectangle',blank)
    cv.waitKey(0)

def circle():
    import cv2 as cv
    import numpy as np
    blank=np.zeros((500,500,3),dtype='uint8')
    cv.circle(blank,(blank.shape[1]//2,blank.shape[0]//2),40,(0,255,0),thickness=-1)
    cv.imshow('Circle',blank)
    cv.waitKey(0)

def line():
    import cv2 as cv
    import numpy as np
    blank=np.zeros((500,500,3),dtype='uint8')
    cv.line(blank,(0,0),(300,400),(255,255,255),thickness=3)
    cv.imshow('Line',blank)
    cv.waitKey(0)

def text():
    import cv2 as cv
    import numpy as np
    blank=np.zeros((500,500,3),dtype='uint8')
    cv.putText(blank,'hello',(0,255),cv.FONT_HERSHEY_TRIPLEX,1.0,(0,255,0),thickness=3)
    cv.imshow('Text',blank)
    cv.waitKey(0)

#Essential Functions
def grayscale():
    import cv2 as cv
    img = cv.imread('Photos/park.jpg')
    cv.imshow('Park', img)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('Gray', gray)
    cv.waitKey(0)

def blur():
    import cv2 as cv
    img = cv.imread('Photos/park.jpg')
    cv.imshow('Park', img)
    blur = cv.GaussianBlur(img, (7,7), cv.BORDER_DEFAULT)
    cv.imshow('Blur', blur)
    cv.waitKey(0)

def edge_cascade():
    import cv2 as cv
    img = cv.imread('Photos/park.jpg')
    cv.imshow('Park', img)
    canny = cv.Canny(blur, 125, 175)
    cv.imshow('Canny Edges', canny)
    cv.waitKey(0)

def dilate():
    import cv2 as cv
    img = cv.imread('Photos/park.jpg')
    cv.imshow('Park', img)
    dilated = cv.dilate(canny, (7,7), iterations=3)
    cv.imshow('Dilated', dilated)
    cv.waitKey(0)

def erode():
    import cv2 as cv
    img = cv.imread('Photos/park.jpg')
    cv.imshow('Park', img)
    eroded = cv.erode(dilated, (7,7), iterations=3)
    cv.imshow('Eroded', eroded)
    cv.waitKey(0)

def resize():
    import cv2 as cv
    img = cv.imread('Photos/park.jpg')
    cv.imshow('Park', img)
    resized = cv.resize(img, (500,500), interpolation=cv.INTER_CUBIC)
    cv.imshow('Resized', resized)
    cv.waitKey(0)

def crop():
    import cv2 as cv
    img = cv.imread('Photos/park.jpg')
    cv.imshow('Park', img)
    cropped = img[50:200, 200:400]
    cv.imshow('Cropped', cropped)
    cv.waitKey(0)

#Transformations
def translate():
    import numpy as np
    import cv2 as cv
    img = cv.imread('Photos/cat.jpg', 0)
    rows, cols = img.shape
    M = np.float32([[1, 0, 100], [0, 1, 50]])
    dst = cv.warpAffine(img, M, (cols, rows))
    cv.imshow('img', dst)
    cv.waitKey(0)
    cv.destroyAllWindows()

def reflection():
    import numpy as np
    import cv2 as cv
    img = cv.imread('Photos/cat.jpg', 0)
    rows, cols = img.shape
    M = np.float32([[1,  0, 0],[0, -1, rows],[0,  0, 1]])
    reflected_img = cv.warpPerspective(img, M,(int(cols),int(rows)))
    cv.imshow('img', reflected_img)
    cv.imwrite('reflection_out.jpg', reflected_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def rotation():
    import numpy as np
    import cv2 as cv
    img = cv.imread('Photos/cat.jpg', 0)
    rows, cols = img.shape
    M = np.float32([[1,  0, 0], [0, -1, rows], [0,  0, 1]])
    img_rotation = cv.warpAffine(img,cv.getRotationMatrix2D((cols/2, rows/2),30, 0.6),(cols, rows))
    cv.imshow('img', img_rotation)
    cv.imwrite('rotation_out.jpg', img_rotation)
    cv.waitKey(0)
    cv.destroyAllWindows()

def scaling():
    import numpy as np
    import cv2 as cv
    img = cv.imread('Photos/cat.jpg', 0)
    rows, cols = img.shape
    img_shrinked = cv.resize(img, (250, 200),interpolation=cv.INTER_AREA)
    cv.imshow('img', img_shrinked)
    img_enlarged = cv.resize(img_shrinked, None,fx=1.5, fy=1.5,interpolation=cv.INTER_CUBIC)
    cv.imshow('img', img_enlarged)
    cv.waitKey(0)
    cv.destroyAllWindows()

def shearing_x_axis():
    import numpy as np
    import cv2 as cv
    img = cv.imread('Photos/cat.jpg', 0)
    rows, cols = img.shape
    M = np.float32([[1, 0.5, 0], [0, 1, 0], [0, 0, 1]])
    sheared_img = cv.warpPerspective(img, M, (int(cols*1.5), int(rows*1.5)))
    cv.imshow('img', sheared_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def shearing_y_axis():
    import numpy as np
    import cv2 as cv
    img = cv.imread('Photos/cat.jpg', 0)
    rows, cols = img.shape
    M = np.float32([[1,   0, 0], [0.5, 1, 0], [0,   0, 1]])
    sheared_img = cv.warpPerspective(img, M, (int(cols*1.5), int(rows*1.5)))
    cv.imshow('sheared_y-axis_out.jpg', sheared_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

#Contours
def contour():
    import cv2 as cv
    import numpy as np
    img=cv.imread('Photos/park.jpg')
    cv.imshow('Park',img)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('Gray', gray)
    edge=cv.Canny(gray,30,300)
    contours,hierachy=cv.findContours(edge,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    cv.imshow('Canny edges after contouring',edge)
    print('No. of contours found: ' + str(len(contours)))
    cv.drawContours(img,contours,-1,(0,255,0),3)
    cv.imshow('Contours',img)
    cv.waitKey(0)
    cv.destroyAllWindows()

#Color Spaces
def color_spaces():
    import cv2 as cv
    img = cv.imread('Photos/park.jpg')
    B, G, R = cv.split(img)

    cv.imshow("original", img)
    cv.waitKey(0)

    cv.imshow("blue", B)
    cv.waitKey(0)
    
    cv.imshow("Green", G)
    cv.waitKey(0)
    
    cv.imshow("red", R)
    cv.waitKey(0)
    
    cv.destroyAllWindows()

#Blurring
def convolutions():
    import cv as cv
    import numpy as np
    # Reading the image
    image = cv.imread('Photos/cats.jpg')
    # Creating the kernel with numpy
    kernel2 = np.ones((5, 5), np.float32)/25
    # Applying the filter
    img = cv.filter2D(src=image, ddepth=-1, kernel=kernel2)
    # showing the image
    cv.imshow('Original', image)
    cv.imshow('Kernel Blur', img)
    cv.waitKey()
    cv.destroyAllWindows()

def averaging():
    import cv2 as cv
    import numpy as np
    # Reading the image
    image = cv.imread('Photos/cats.jpg')
    # Applying the filter
    averageBlur = cv.blur(image, (5, 5))
    # Showing the image
    cv.imshow('Original', image)
    cv.imshow('Average blur', averageBlur)
    cv.waitKey()
    cv.destroyAllWindows()

def gaussian():
    import cv2 as cv
    import numpy as np
    # Reading the image
    image = cv.imread('Photos/cats.jpg')
    # Applying the filter
    gaussian = cv.GaussianBlur(image, (3, 3), 0)
    # Showing the image
    cv.imshow('Original', image)
    cv.imshow('Gaussian blur', gaussian)
    cv.waitKey()
    cv.destroyAllWindows()

def median():
    import cv2 as cv
    import numpy as np
    # Reading the image
    image = cv.imread('Photos/cats.jpg')
    # Applying the filter
    medianBlur = cv.medianBlur(image, 9)
    cv.medianBlur(image, 9)
    # Showing the image
    cv.imshow('Original', image)
    cv.imshow('Median blur', 
    medianBlur)
    cv.waitKey()
    cv.destroyAllWindows()

def bilateral():
    import cv2 as cv
    import numpy as np
    # Reading the image
    image = cv.imread('Photos/cats.jpg')
    # Applying the filter
    bilateral = cv.bilateralFilter(image, 
    9, 75, 75)
    # Showing the image
    cv.imshow('Original', image)
    cv.imshow('Bilateral blur', bilateral)
    cv.waitKey()
    cv.destroyAllWindows()

#Bitwise Operations
def bitwise_and_():
    import cv2 as cv
    import numpy as np
    blank = np.zeros((400,400), dtype='uint8')
    rectangle = cv.rectangle(blank.copy(), (30,30), (370,370), 255, -1)
    circle = cv.circle(blank.copy(), (200,200), 200, 255, -1)
    cv.imshow('Rectangle', rectangle)
    cv.imshow('Circle', circle)
    bitwise_and = cv.bitwise_and(rectangle, circle)
    cv.imshow('Bitwise AND', bitwise_and)
    cv.waitKey(0)

def bitwise_or_():
    import cv2 as cv
    import numpy as np
    blank = np.zeros((400,400), dtype='uint8')
    rectangle = cv.rectangle(blank.copy(), (30,30), (370,370), 255, -1)
    circle = cv.circle(blank.copy(), (200,200), 200, 255, -1)
    cv.imshow('Rectangle', rectangle)
    cv.imshow('Circle', circle)
    bitwise_or = cv.bitwise_or(rectangle, circle)
    cv.imshow('Bitwise OR', bitwise_or)
    cv.waitKey(0)

def bitwise_xor_():
    import cv2 as cv
    import numpy as np
    blank = np.zeros((400,400), dtype='uint8')
    rectangle = cv.rectangle(blank.copy(), (30,30), (370,370), 255, -1)
    circle = cv.circle(blank.copy(), (200,200), 200, 255, -1)
    cv.imshow('Rectangle', rectangle)
    cv.imshow('Circle', circle)
    bitwise_xor = cv.bitwise_xor(rectangle, circle)
    cv.imshow('Bitwise XOR', bitwise_xor)
    cv.waitKey(0)

def bitwise_not_():
    import cv2 as cv
    import numpy as np
    blank = np.zeros((400,400), dtype='uint8')
    rectangle = cv.rectangle(blank.copy(), (30,30), (370,370), 255, -1)
    circle = cv.circle(blank.copy(), (200,200), 200, 255, -1)
    cv.imshow('Rectangle', rectangle)
    cv.imshow('Circle', circle)
    bitwise_not = cv.bitwise_not(circle)
    cv.imshow('Circle NOT', bitwise_not)
    cv.waitKey(0)

#Masking
def mask():
    import cv2 as cv
    import numpy as np
    img = cv.imread('Photos/cats.jpg')
    cv.imshow('Original image', img)
    blank = np.zeros(img.shape[:2], dtype='uint8')
    cv.imshow('Blank Image', blank)
    circle = cv.circle(blank,
    (img.shape[1]//2,img.shape[0]//2),200,255, -1)
    cv.imshow('Mask',circle)
    masked = cv.bitwise_and(img,img,mask=circle)
    cv.imshow('Masked Image', masked)
    cv.waitKey(0)

def alpha_blending():
    import cv2
    img1 = cv2.imread('Photos/cats2.jpg')
    img2 = cv2.imread('Photos/cat.jpg')
    cv2.imshow("img 1",img1)
    cv2.waitKey(0)
    cv2.imshow("img 2",img2)
    cv2.waitKey(0)
    choice = 1
    while (choice) :
        alpha = float(input("Enter alpha value"))
        dst = cv2.addWeighted(img1, alpha , img2, 
        1-alpha, 0)
        cv2.imwrite('alpha_mask_.png', dst)
        img3 = cv2.imread('alpha_mask_.png')
        cv2.imshow("alpha blending 1",img3)
        cv2.waitKey(0)
        choice = int(input("Enter 1 to continue and 0 to exit"))

#Histogram
def histogram():
    import cv2 as cv
    from matplotlib import pyplot as plt
    img = cv.imread('Photos/park.jpg',0)
    histr = cv.calcHist([img],[0],None,[256],[0,256])
    plt.plot(histr)
    plt.show()
