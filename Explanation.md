# What is a module?
A module is essentially a library with organised code. This code is useful for performing specific tasks. To use a module, you have to install or download it. In this case, the module is custom-made. So, it should be downloaded.

# How to use my custom module?
A custom module should be downloaded to a folder where you are going to put your project files. In this case, make sure to download the Photos and Videos folder as well, because it contains resources which will be used in the module.

# How does the module work?
In the module is a bunch of defined functions with code inside each one of them to perform specific tasks. For instance, resize_image() is a function with code that resizes the given image. If you want to use this or any other function, you have to type the following in your project file:

                                    from module.py import function_name
                                    function_name()
Make sure that you replace the function_name with the actual name of the function. If you want to, for instance, resize the image, then you can do following:

                                    Eg:
                                    from module.py import resize_image
                                    resize_image()

# Important points to remember while coding with the module:
1. If the error '-215 Assertion failed' arises, it can be for two reasons:
   1. If you are reading a video using the cv.imread() function and the error arises after the video is done playing, then there is nothing with it as it is just telling that it can't play the video anymore due to the frames being done.
   2. If the error comes before the video starts playing or the image gets read and this error doesn't let the image open or the video play, then it means that you have typed the path of the file or the location of file wrong. To fix this, check if you have made any mistakes while typing the location of the file inside cv.imread("").
2. If the error 'matplotlib is not defined' arises, then you have to install it in command prompt, just like how you install opencv and caer. The code to install is:

                                    pip install matplotlib
3. If any other error arises, the line with the error will show in the terminal. Kindly go to specified line and check that line and a few more lines before it for any spelling mistakes or syntax errors.
