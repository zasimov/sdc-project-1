import os

import cv2
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np

import math

from collections import namedtuple

import metrics
from helpers import show_images


# type to describe colors in HLS color space
HLS = namedtuple('HLS', ['h', 'l', 's'])
RGB = namedtuple('RGB', ['r', 'g', 'b'])


class Glass:
    """Glass adjusts colors between lower and upper

    lower and upper must be a color in the same color space as image

    __call__ returns reflected image.
    """
    
    def __init__(self, lower, upper):
        self._lower = lower
        self._upper = upper

    def __call__(self, image):
        return cv2.inRange(image, self._lower, self._upper)


# that's very important parameters
hls_yellow_glass = Glass(HLS(10, 0, 100), HLS(40, 255, 255))
hls_white_glass = Glass(HLS(0, 200, 0), HLS(255, 255, 255))

rgb_yellow_glass = Glass(RGB(180, 180, 0), RGB(255, 255, 255))
rgb_white_glass = Glass(RGB(200, 200, 200), RGB(255, 255, 255))


def mix(image1, image2):
    return cv2.bitwise_or(image1, image2)


def apply_mask(image, mask):
    return cv2.bitwise_and(image, image, mask=mask)


def select_yellow_and_white(image):
    hls_image = hls(image)
    yellow = hls_yellow_glass(hls_image)
    white = hls_white_glass(hls_image)
    mask = mix(yellow, white)
    return apply_mask(image, mask)


def select_yellow_and_white_rgb(image):
    yellow = rgb_yellow_glass(image)
    white = rgb_white_glass(image)
    mask = mix(yellow, white)
    return apply_mask(image, mask)


# i hate dots
# and i like octave functions =)
imread = mpimg.imread
imshow = plt.imshow
subplot = plt.subplot
figure = plt.figure
show = plt.show
title = plt.title
tight_layout = plt.tight_layout
xticks = plt.xticks
yticks = plt.yticks

pi = math.pi

import math
sqrt = math.sqrt


def show_images(images, c=2):
    """show_images shows a list of images

    Arguments:
      - c - a number of columns
    """
    r = len(images) // 2

    figure()
    for n, pair in enumerate(images):
        image_name, image = pair
        subplot(r, c, n + 1)
        imshow(image)
        title(image_name)
        xticks(())
        yticks(())
    tight_layout(pad=0, h_pad=0, w_pad=0)
    show()


def is_not_pseudo_dir(dirname):
    return dirname not in [".", ".."]

def hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)


def hls(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)


def rgb2gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def grayshow(gray):
    imshow(gray, cmap='gray')


def gauss(image, n, sigma):
    return cv2.GaussianBlur(image,(n, n), sigma)
    

def canny(gray, low_threshold, high_threshold, N, sigma):
    smoothed = gauss(gray, N, sigma)
    return cv2.Canny(smoothed, low_threshold, high_threshold)


def black(image):
    return np.copy(image) * 0


def hough(edges, rho, theta, threshold, min_line_length, max_line_gap):
    return cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                           min_line_length, max_line_gap)


def draw_lines(img, lines, color=[255, 0, 0], thickness=4):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            

def color_edges(edges):
    # Create a "color" binary image to combine with line image
    return np.dstack((edges, edges, edges))


def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


def trapeze(image, bottom_width, top_width):
    height = image.shape[0]
    width = image.shape[1]

    x_middle = width / 2
    y_middle = height / 2

    vertices = [[
        (bottom_width / 2, height),
        (x_middle - top_width / 2, y_middle + top_width / 2), 
        (x_middle + top_width / 2, y_middle + top_width / 2),
        (width - bottom_width / 2, height)
    ]]
    
    return np.array(vertices, dtype=np.int32)


def region(edges, bottom_width, top_width, ignore_mask_color=255):
    mask = np.zeros_like(edges)
    imshape = edges.shape
    vertices = trapeze(edges, bottom_width, top_width)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    return cv2.bitwise_and(edges, mask)


test_images = os.listdir("test_images")
test_images = filter(is_not_pseudo_dir, test_images)
test_images = list(sorted(test_images))

test_images = [(image_name, imread(os.path.join("test_images", image_name))) for image_name in test_images]

#show_images(test_images)

Configuration = namedtuple('Configuration', [
    'low_t',
    'high_t',
    'N',
    'sigma',
    'top_width',
    'bottom_width',
    'rho',
    'theta',
    'threshold',
    'min_line_length',
    'max_line_gap'
    ])


c = 2
r = len(test_images) // 2
n = 1


class SlopeInterceptLine:

    def __init__(self, slope, intercept):
        assert slope != 0
        self.slope = slope
        self.intercept = intercept

    def fit(self, image, horizon=0.6):
        """Returns a line segment from bottom of image to horizon"""
        y1 = image.shape[0]
        y2 = y1 * horizon

        x1 = int((y1 - self.intercept) / self.slope)
        x2 = int((y2 - self.intercept) / self.slope)
        y1 = int(y1)
        y2 = int(y2)

        coords = np.array((x1, y1, x2, y2))
        
        return Segment(coords)


class Segment:
    
    def __init__(self, coords):
        self._coords = coords   

    @property
    def x1(self):
        return self._coords[0]

    @property
    def y1(self):
        return self._coords[1]

    @property
    def x2(self):
        return self._coords[2]

    @property
    def y2(self):
        return self._coords[3]

    @property
    def is_vertical(self):
        return self.x1 == self.x2

    @property
    def slope(self):
        return (self.y2 - self.y1) / (self.x2 - self.x1)

    @property
    def intercept(self):
        return self.y1 - self.slope * self.x1

    @property
    def length(self):
        return np.sqrt((self.y2 - self.y1)**2 + (self.x2 - self.x1)**2)

    def to_tuple(self):
        return tuple(self._coords)

    @property
    def stats(self):
        return (self.slope, self.intercept, self.length)
        

def without_vertical(lines):
    """Returns only reasonable lines

    1. Vertical lines is not reasonable

    Also lines processor wraps each line to Line class
    """
    for line in lines:
        line_obj = Segment(line[0])
        if line_obj.is_vertical:
            continue # just drop all vertical lines
        yield line_obj


from operator import attrgetter


class Candidates:
    
    def __init__(self, lines):
        self.lines = lines

    def slope_moments(self):
        slopes = list(map(attrgetter('slope'), self.lines))
        return metrics.RandomVarMoments(np.mean(slopes), np.std(slopes))

    def intercept_moments(self):
        intercepts = list(map(attrgetter('intercept'), self.lines))
        return metrics.RandomVarMoments(np.mean(intercepts), np.std(intercepts))

    def length_moments(self):
        lengths = list(map(attrgetter('length'), self.lines))
        return metrics.RandomVarMoments(np.mean(lengths), np.std(lengths))

    def append(self, line):
        self.lines.append(line)

    @property
    def is_empty(self):
        return not bool(self.lines)

    def __iter__(self):
        return iter(self.lines)
    

class Lane:
    def __init__(self, left, right):
        self.left = left
        self.right = right

    @property
    def found(self):
        return bool(self.left and self.right)


class LaneDetector:
    """LaneDetector is an object that detects left and right lane

    Can have state
    """

    def __call__(self, lines):
        """Lane detector uses lines_processor to filter lines and classifies lines to "left" and "right"
        
        Returns Lane object
        """
        left_lines = []
        right_lines = []
    
        # drop vertical lines and split lines to "left" and "right"                                     
        for line in without_vertical(lines):
            if line.slope < 0:
                # left line
                left_lines.append(line)
            elif line.slope > 0:
                # right line
                right_lines.append(line)
    
        left_candidates = Candidates(left_lines) if left_lines else None
        right_candidates = Candidates(right_lines) if right_lines else None
    
        return Lane(left_candidates, right_candidates)


from collections import deque


class Stabilizer:
    """Stabilizes Lane"""

    def __init__(self, memory_size=20):
        self.left_candidates = deque(maxlen=memory_size)
        self.right_candidates = deque(maxlen=memory_size)

    def mean_line(self, candidates):
        if not candidates:
            return None
        
        mean_slope = np.mean(list(map(attrgetter("slope"), candidates)))
        mean_intercept = np.mean(list(map(attrgetter("intercept"), candidates)))

        return SlopeInterceptLine(mean_slope, mean_intercept)

    def __call__(self, lane_candidate):
        left_candidates = lane_candidate.left
        right_candidates = lane_candidate.right

        left_mean_line = self.mean_line(left_candidates)
        if left_mean_line:
            self.left_candidates.append(left_mean_line)

        right_mean_line = self.mean_line(right_candidates)
        if right_mean_line:
            self.right_candidates.append(right_mean_line)

        return Lane(self.mean_line(self.left_candidates),
                    self.mean_line(self.right_candidates))
    

class Pipeline:
    
    def __init__(self, configuration):
        self.configuration = configuration
        # stores per-frame metrics
        self.metrics = []
        self.lane_detector = LaneDetector()
        self.stabilizer = Stabilizer()

    def _draw_lane(self, image, lane):
        left = lane.left.fit(image)
        right = lane.right.fit(image)
        draw_lines(image, [[left.to_tuple()], [right.to_tuple()]])

    def _draw_machine_view(self, image, machine_view):
        ratio = image.shape[0] / image.shape[1]
        small_width = 300
        small_height = int(small_width * ratio)
        small_machine_view = cv2.resize(machine_view, (small_width, small_height))
        image[20:(20 + small_height),20:(20 + small_width),:] = small_machine_view

    def _color_selection(self, image):
        return select_yellow_and_white(image)

    def __call__(self, input_image):
        frame_metrics = metrics.FrameMetrics()

        machine_view = np.copy(input_image)
        machine_view = self._color_selection(machine_view)
        
        image = self._color_selection(input_image)
        gray = rgb2gray(image)
        edges = canny(gray, self.configuration.low_t, self.configuration.high_t,
                      self.configuration.N, self.configuration.sigma)
        edges = region(edges, self.configuration.bottom_width, self.configuration.top_width)
        lines =  hough(edges, self.configuration.rho, self.configuration.theta, self.configuration.threshold,
                       self.configuration.min_line_length, self.configuration.max_line_gap)
        
        if lines is None:
            lines = []

        frame_metrics['line_count'] = len(lines)

        draw_lines(machine_view, lines, color=[127, 127, 127])

        lane = self.lane_detector(lines)
        metrics.collect_candidates(frame_metrics, lane)

        lane = self.stabilizer(lane)

        if lane.found:
            frame_metrics['missed'] = 0
        else:
            frame_metrics['missed'] = 1

        if lane.found:
            self._draw_lane(input_image, lane)
            self._draw_lane(machine_view, lane)

            self._draw_machine_view(input_image, machine_view)

            left = lane.left.fit(image)
            right = lane.right.fit(image)
            frame_metrics.collect_line(metrics.LEFT, left)
            frame_metrics.collect_line(metrics.RIGHT, right)        

        self.metrics.append(frame_metrics)

        return input_image


class RGBPipeline(Pipeline):

    def _color_selection(self, image):
        return select_yellow_and_white_rgb(image)


from moviepy.editor import VideoFileClip


import contextlib


def test(pipeline_cls, configuration, output_subdir, stats_file):
    all_metrics = metrics.Metrics()

    for clip in sorted(os.listdir("test_videos")):
        input_ = os.path.join('test_videos', clip)
        output_path = os.path.join('output', output_subdir)
        with contextlib.suppress(FileExistsError):
            os.makedirs(output_path)
        output = os.path.join(output_path, clip)
        clip1 = VideoFileClip(input_)
        pipeline = pipeline_cls(configuration)
        white_clip = clip1.fl_image(pipeline)
        all_metrics.collect(clip, pipeline.metrics)        
        white_clip.write_videofile(output, audio=False)

    all_metrics.save(stats_file)


winner = Configuration(
    low_t=20,
    high_t=100,
    N=15,
    sigma=0,
    top_width=80,
    bottom_width=150,
    rho=2,
    theta=pi / 180,
    threshold=20,
    min_line_length=20,
    max_line_gap=400)


wrong_canny = Configuration(
    low_t=100,
    high_t=300,
    N=15,
    sigma=0,
    top_width=80,
    bottom_width=150,
    rho=2,
    theta=pi / 180,
    threshold=20,
    min_line_length=20,
    max_line_gap=400)


wrong_hough = Configuration(
    low_t=20,
    high_t=100,
    N=15,
    sigma=0,
    top_width=80,
    bottom_width=150,
    rho=2,
    theta=pi / 180,
    threshold=10,
    min_line_length=10,
    max_line_gap=40)

                    
processed = [(image_name, Pipeline(winner)(image)) for (image_name, image) in test_images]
# remove comment if you want to see test images
#show_images(processed)


test(Pipeline, winner, 'winner', 'stats.csv')
test(RGBPipeline, winner, 'rgb', 'rgb_stats.csv')
test(Pipeline, wrong_canny, 'wrong_canny', 'wrong_canny_stats.csv')
test(Pipeline, wrong_hough, 'wrong_hough', 'wrong_hough_stats.csv')
