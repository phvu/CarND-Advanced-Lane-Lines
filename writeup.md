## Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistorted.png "Undistorted"
[image2]: ./output_images/binarize.png "Binarized images"
[image3]: ./output_images/warp.png "Warping"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in lines 57 through 84 of the file called `detect_lanes.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. 
Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points 
are the same for each calibration image.  
Thus, `objp` is just a replicated array of coordinates, and `objpoints` 
will be appended with a copy of it every time I successfully detect all 
chessboard corners in a test image.  
`imgpoints` will be appended with the (x, y) pixel position of each of the 
corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera 
calibration and distortion coefficients using the `cv2.calibrateCamera()` function. 
 I applied this distortion correction to the test image using the 
 `cv2.undistort()` function and obtained this result: 

![Undistored images][image1]

To produce  the above image: `python detect_lanes_test.py DetectLanesTest.test_undistort`

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![Undistored images][image1]

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 94 through 132 in `detect_lanes.py`).

The image is processed in the HLS space. The L channel is first went through `Contrast Limited Adaptive Histogram Equalization`
so that the overall contrast is improved. Note that in the real pipeline, the input 
of this step is the warped image of the road, so I thought maintaining a good contrast
is important for detecting the lines.

The L channel is then blurred with a Gaussian filter, and then a process similar to
[image sharpening](http://stackoverflow.com/questions/19890054/how-to-sharpen-an-image-in-opencv),
however by using weights of 1 and -1, I was able to skip irrelevant details.
I found this technique is capable of detecting the while lane in the middle of the road,
and with proper thresholds, it is more robust than gradient threshold.

As suggested in the lecture, the S channel also goes through some thresholds, and
the results are combined. 

Before thresholding, I realized it is important to normalize the channels using 
the `cv2.normalize()` function.

The S channel is a bit sensitive to shadows on the road, where it will create big blobs.
To remove the blobs, I discard all rows in the binary image that have more than 
half of the pixels activated (lines 130-131). This is a bit naive though, we probably
can do it more robustly by applying other techniques like finding the contours,
or watershed, and then shrinking the big blobs. Nevertheless, the simple technique
seems to work for the project video and the challenge video.

Here's an example of my output for this step, produced by `python detect_lanes_test.py DetectLanesTest.test_binarize`

![alt text][image2]

Note that in the real pipeline, we feed the warped image into this function,
so the trees and everything above that will be left out automatically.
It's important that the two lines are detected reliably.

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`,
which appears in lines 136 through 152 in the file `detect_lanes.py`.

I chose the hardcode the source and destination points in the following manner:

```
src = np.float32([
        [595, 450],
        [685, 450],
        [1040, 674],
        [269, 674]])
    dst = np.float32([
        [200, 10],
        [1000, 10],
        [1000, 700],
        [200, 700]])
```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 595, 450      | 200, 10       | 
| 685, 450      | 1000, 10      |
| 1040, 674     | 1000, 700     |
| 269, 674      | 200, 700      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a 
test image and its warped counterpart to verify that the lines appear parallel 
in the warped image.

![alt text][image3]

Image produced by `python detect_lanes_test.py DetectLanesTest.test_warp`

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

