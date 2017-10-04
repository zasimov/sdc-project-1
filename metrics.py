"""
Module contains classes and functions to collect Lane detector metrics
"""

import csv

from collections import namedtuple


FRAME_METRICS = ['line_count',
                 # left lane candidates metrics
                 'left_slope_mean', 'left_slope_std',
                 'left_intercept_mean', 'left_intercept_std',
                 'left_length_mean', 'left_length_std',
                 # right lane candidates metrics
                 'right_slope_mean', 'right_slope_std',
                 'right_intercept_mean', 'right_intercept_std',
                 'right_length_mean', 'right_length_std',
                 # left lane metrics
                 'left_lane_slope', 'left_lane_intercept', 'left_lane_length',
                 # right lane metrics
                 'right_lane_slope', 'right_lane_intercept', 'right_lane_length',
                 'missed'
]


LEFT = 'left'
RIGHT = 'right'


class FrameMetrics(dict):

    def collect_line(self, name, line):
        slope, intercept, length = line.stats
        self['%s_lane_slope' % name] = slope
        self['%s_lane_intercept' % name] = intercept
        self['%s_lane_length' % name] = length
    
    def collect_moments(self, moments_name, moments):
        """Save moments

        For example:

          frame_metrics.set_moments('left_slope', candidates.slope_moments())
        """
        self['%s_mean' % moments_name] = moments.mean
        self['%s_std' % moments_name] = moments.std


def collect_candidates(frame_metrics, lane):
    if lane.left:
        frame_metrics.collect_moments('left_slope', lane.left.slope_moments())
        frame_metrics.collect_moments('left_intercept', lane.left.intercept_moments())
        frame_metrics.collect_moments('left_length', lane.left.length_moments())

    if lane.right:
        frame_metrics.collect_moments('right_slope', lane.right.slope_moments())
        frame_metrics.collect_moments('right_intercept', lane.right.intercept_moments())
        frame_metrics.collect_moments('right_length', lane.right.length_moments())
        

# data structure to store moments of Random Variable
RandomVarMoments = namedtuple('RandomVarMoments', ['mean', 'std'])


class Metrics:
    """collects metrics for each clip

    You can save metrics to file using "save" method
    """
    
    def __init__(self):
        # map from "clip name" to metrics
        self._metrics = {}

    def collect(self, clip_name, metrics):
        """Save clip metrics

        * clip_name - a file name
        * metrics - a list of per-frame metrics
        """
        self._metrics[clip_name] = metrics

    def save(self, filename):
        with open(filename, 'w') as out:
            header = ['clip', 'clipno', 'frame'] + FRAME_METRICS
            
            writer = csv.DictWriter(fieldnames=header, f=out)
            writer.writeheader()

            clipno = 1

            for clip in sorted(self._metrics.keys()):
                metrics = self._metrics[clip]
                for frame, frame_metrics in enumerate(metrics):
                    row = dict(clip=clip, clipno=clipno, frame=frame)
                    row_stats = {k: frame_metrics.get(k, 0.0) for k in FRAME_METRICS}
                    row.update(row_stats)
                    writer.writerow(row)
                clipno += 1
