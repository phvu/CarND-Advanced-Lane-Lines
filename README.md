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
[image4]: ./output_images/preprocess.png "Preprocess"
[image5]: ./output_images/detect_lanes.png "Fit Visual"
[image6]: ./output_images/draw_lane.png "Output"

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

Combining distortion, warping and binarize, I have a binary mask of road image like this
(produced by `python detect_lanes_test.py DetectLanesTest.test_preprocess`)

![alt text][image4]

With those binary mask, using the code similar to what provided in the lectures,
(in the `detect_lanes()` and `infer_lanes()` functions from lines 
155 to 294 of `detect_lanes.py`), I was able to detect the lanes that look like this:
 
![alt text][image5]

Image produced with `python detect_lanes_test.py DetectLanesTest.test_detect_lanes`

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The curvature is computed in `compute_curvature()`, lines 297-314 of `detect_lanes.py`.
The idea is to map the location of the left and right fits to the real world coordinates,
then take the radius of the curvature in the real-world coordinates.

The position of the car is defined as `(center_of_image - center_of_lane) * xm_per_pixel`.
`center_of_image` and `xm_per_pixel` are easy to compute (or to guess).
To compute `center_of_lane`, I projected the coordinates of the fitted lines
in the warped image back to the camera coordinate. It is simply a matrix
multiplication with the inverse of the perspective transformation matrix.

This is implemented in `compute_position()`, lines 317-323 of `detect_lanes.py`.
The result is negative if the car is on the left side and positive if the car
is on the right side compared to the middle of the lane.

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 326 through 337 in my code in `detect_lanes.py` 
in the function `draw_lane()`. This function only draw the lane in the warped
coordinates. I then unwarp it using the inverse matrix and add this layer onto
the original image (lines 374-375 of `detect_lanes.py`)

Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

This is my result for the project video, can be re-produced with 
`python detect_lanes.py project_video.mp4 -d`.

The output file is `project_video_detected.mp4`. The left panel
can be disable by removing the `-d` option when running `detect_lanes.py`.

[![Project video](https://img.youtube.com/vi/ZJUO1hmXBTg/0.jpg)](https://youtu.be/ZJUO1hmXBTg)

This is the result for the challenge video, produced with `python detect_lanes.py challenge_video.mp4 -d`

[![Project video](https://img.youtube.com/vi/0tUX-tZPkl8/0.jpg)](https://youtu.be/0tUX-tZPkl8)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Making the code works for `project_video.mp4` is pretty easy. I did the following
to make it (somewhat) works for the challenge video:

1. Thresholding on the L channel, instead of gradient thresholding, as discussed in the `Pipeline (single image)` section.
2. Improve the inference of lane lines when the vehicle is turning:
    
    This is done in line 251-257:
    
        margin = 50
        left_line = left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2]
        right_line = right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2]
        left_lane_inds = ((nonzerox > (left_line - ((2 if left_fit[0] > 0 else 1) * margin))) &
                          (nonzerox < (left_line + ((2 if left_fit[0] < 0 else 1) * margin))))
        right_lane_inds = ((nonzerox > (right_line - ((2 if right_fit[0] > 0 else 1) * margin))) &
                           (nonzerox < (right_line + ((2 if right_fit[0] < 0 else 1) * margin))))
                           
    The key idea is that if the car is turning left (`left_fit[0] > 0`), then we will
    extend the search area on the left, and vice versa.
    
    This help the pipeline works for the challenge video, although it failed a bit 
    toward the end of that video.
    
    I see the following improvement that should've been done (probably for future work):
    
    1. Some kind of smoothing is needed to make sure the lines are not changing
    dramatically from one frame to the next.
    2. Sanity check the quality of the detected lane. I tried to do some of this
    in lines 33-54, but some more could be done.
    