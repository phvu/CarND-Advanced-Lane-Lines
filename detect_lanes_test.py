import unittest

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import VideoFileClip

import detect_lanes


class DetectLanesTest(unittest.TestCase):

    @staticmethod
    def run_video(input_video, output_video, f):
        """
        Run the function f on every frame of input_video, write the result to output_video\
        """
        out_clip = VideoFileClip(input_video).fl_image(f)
        out_clip.write_videofile(output_video, audio=False, verbose=False)

    @classmethod
    def setUpClass(cls):
        cls.mtx, cls.dist = detect_lanes.calibrate()
        cls.transform_matrix = detect_lanes.get_transform()
        cls.lane = None

    def test_binarize_video(self):

        def binarize_func(img):
            img = img[:, :, ::-1]
            undistorted = detect_lanes.undistort(img, self.mtx, self.dist)
            warped = detect_lanes.warp(self.transform_matrix, undistorted)
            hls = cv2.cvtColor(warped, cv2.COLOR_BGR2HLS)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            hls[:, :, 1] = clahe.apply(hls[:, :, 1])

            l_image = hls[:, :, 1]
            l_blur = cv2.GaussianBlur(l_image, (0, 0), 9)
            l_image = cv2.addWeighted(l_image, 1, l_blur, -1, 0)
            l_image = cv2.normalize(l_image, np.zeros_like(l_image), 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
            l_binary = np.zeros_like(l_image)
            l_binary[(l_image >= 50) & (l_image <= 255)] = 1

            s_channel = hls[:, :, 2]
            s_binary = np.zeros_like(s_channel)
            s_binary[(s_channel >= 100) & (s_channel <= 255)] = 1

            combined_binary = np.zeros_like(l_binary)
            combined_binary[(s_binary == 1) | (l_binary == 1)] = 1

            if self.lane is None:
                self.lane, _ = detect_lanes.detect_lanes(combined_binary)
            else:
                self.lane = detect_lanes.infer_lanes(combined_binary, self.lane.left_fit, self.lane.right_fit)

            out = detect_lanes.draw_lane(self.lane.left_fitx, self.lane.right_fitx, combined_binary)
            masks = np.dstack((combined_binary * 255, combined_binary * 255, combined_binary * 255))
            return np.hstack((img[:, :, ::-1], cv2.addWeighted(masks, 1, out, 0.6, 0)))

        DetectLanesTest.run_video('challenge_video.mp4', 'output_images/challenge_video_binarize.mp4', binarize_func)

    def test_equalize_hist(self):

        def equalize(img):
            hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            # img2 = cv2.cvtColor(hls, cv2.COLOR_HLS2BGR)
            # b = binarize(img2, s_thres=(20, 255)) * 255
            # binarize_img = np.dstack((b, np.zeros_like(b), np.zeros_like(b)))
            s_image = hls[:, :, 1:].sum(2).astype(np.float64) / 2
            s_image = (s_image - s_image.min()) / (s_image.max() - s_image.min())
            s_image = (s_image * 255).astype(np.uint8)
            s_image = np.dstack((s_image, np.zeros_like(s_image), np.zeros_like(s_image)))
            return np.hstack((img, s_image))

        DetectLanesTest.run_video('challenge_video.mp4', 'output_images/challenge_video_histogram.mp4', equalize)

    def test_undistort(self):
        files = ['camera_cal/calibration1.jpg', 'camera_cal/calibration9.jpg',
                 'test_images/test4.jpg', 'test_images/test5.jpg']

        gs = mpl.gridspec.GridSpec(len(files), 2)
        gs.update(wspace=0.01, hspace=0.01, left=0.1, right=0.9, bottom=0.1, top=0.9)
        plt.figure(figsize=(11, 3))
        for idx, fname in enumerate(files):
            test_img = cv2.imread(fname)
            test_img_undistort = detect_lanes.undistort(test_img, self.mtx, self.dist)
            plt.subplot(gs[2*idx])
            plt.imshow(test_img[:, :, ::-1])
            plt.axis('off')
            plt.subplot(gs[2*idx + 1])
            plt.imshow(test_img_undistort[:, :, ::-1])
            plt.axis('off')
        plt.show()

    def test_binarize(self):
        files = ['test_images/test1.jpg', 'test_images/test2.jpg',
                 'test_images/test4.jpg', 'test_images/test5.jpg']

        gs = mpl.gridspec.GridSpec(len(files), 2)
        gs.update(wspace=0.01, hspace=0.01, left=0.1, right=0.9, bottom=0.1, top=0.9)
        plt.figure(figsize=(11, 3))
        for idx, fname in enumerate(files):
            test_img = cv2.imread(fname)
            test_img_undistort = detect_lanes.undistort(test_img, self.mtx, self.dist)
            test_binary = detect_lanes.binarize(test_img_undistort, s_thres=(100, 255), l_thres=(50, 255))

            plt.subplot(gs[2 * idx])
            plt.imshow(test_img[:, :, ::-1])
            plt.axis('off')
            plt.subplot(gs[2 * idx + 1])
            plt.imshow(test_binary, cmap='gray')
            plt.axis('off')
        plt.show()

    def test_warp(self):
        files = ['test_images/straight_lines1.jpg', 'test_images/test1.jpg', 'test_images/test3.jpg']
        gs = mpl.gridspec.GridSpec(len(files), 2)
        gs.update(wspace=0.01, hspace=0.01, left=0.1, right=0.9, bottom=0.1, top=0.9)
        plt.figure(figsize=(11, 3))

        for idx, fname in enumerate(files):
            test_img = cv2.imread(fname)
            test_img_warp = detect_lanes.warp(self.transform_matrix, test_img)

            if idx == 0:
                for (x, y) in [[595, 450], [685, 450], [1040, 674], [269, 674]]:
                    cv2.circle(test_img, (x, y), 10, (255, 0, 0), -1)
                for (x, y) in [[200, 10], [1000, 10], [1000, 700], [200, 700]]:
                    cv2.circle(test_img_warp, (x, y), 10, (0, 255, 0), -1)

            plt.subplot(gs[2 * idx])
            plt.imshow(test_img[:, :, ::-1])
            plt.axis('off')
            plt.subplot(gs[2 * idx + 1])
            plt.imshow(test_img_warp[:, :, ::-1])
            plt.axis('off')
        plt.show()

    def test_preprocess(self):
        files = ['test_images/test1.jpg', 'test_images/test2.jpg',
                 'test_images/test4.jpg', 'test_images/test5.jpg']

        gs = mpl.gridspec.GridSpec(len(files), 2)
        gs.update(wspace=0.01, hspace=0.01, left=0.1, right=0.9, bottom=0.1, top=0.9)
        plt.figure(figsize=(11, 3))
        for idx, fname in enumerate(files):
            test_img = cv2.imread(fname)
            test_img_undistort = detect_lanes.undistort(test_img, self.mtx, self.dist)
            warped = detect_lanes.warp(self.transform_matrix, test_img_undistort)
            test_binary = detect_lanes.binarize(warped, s_thres=(100, 255), l_thres=(50, 255))

            plt.subplot(gs[2 * idx])
            plt.imshow(test_img[:, :, ::-1])
            plt.axis('off')
            plt.subplot(gs[2 * idx + 1])
            plt.imshow(test_binary, cmap='gray')
            plt.axis('off')
        plt.show()

    def test_detect_lanes(self):
        files = ['test_images/test1.jpg', 'test_images/test2.jpg', 'test_images/test4.jpg']

        gs = mpl.gridspec.GridSpec(len(files), 2)
        gs.update(wspace=0.01, hspace=0.01, left=0.1, right=0.9, bottom=0.1, top=0.9)
        plt.figure(figsize=(11, 3))
        for idx, fname in enumerate(files):
            test_img = cv2.imread(fname)
            test_img_undistort = detect_lanes.undistort(test_img, self.mtx, self.dist)
            warped = detect_lanes.warp(self.transform_matrix, test_img_undistort)
            test_binary = detect_lanes.binarize(warped, s_thres=(100, 255), l_thres=(50, 255))

            lane, vis_img = detect_lanes.detect_lanes(test_binary)
            lane_img = detect_lanes.draw_lane(lane.left_fitx, lane.right_fitx, test_binary)
            # white_mask = np.dstack((test_binary * 255, test_binary * 255, test_binary * 255))
            lane_img = cv2.addWeighted(vis_img, 1, lane_img, 0.6, 0)

            plt.subplot(gs[2 * idx])
            plt.imshow(test_img[:, :, ::-1])
            plt.axis('off')
            plt.subplot(gs[2 * idx + 1])
            plt.imshow(lane_img)
            plt.axis('off')
        plt.show()

if __name__ == '__main__':
    unittest.main()
