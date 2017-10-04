## load_stats and load_rgb_stats should be loaded
## and load_wrong_canny_stats ...
## and load_wrong_hough_stats

## This script prepares illustrations for write up.

close all;

challenge = clip1;


##
## HLS vs RGB (line count)
##
p = figure();
hold on;

plot(line_count(challenge), 'r', rgb_line_count(challenge))
legend('hls color selection', 'rgb color selection')
title('HLS vs RGB')
xlabel("frame #")
ylabel("line count")

print(p, "illustrations/hls_vs_rgb.png", '-dpng');

hold off;


##
## How do Canny thresholds affect line count
##
p = figure();
hold on;

plot(line_count(challenge), 'r', wrong_canny_line_count(challenge))
legend('low=20, high=100', 'low=100, high=300')
title('Line count and Canny thresholds')
xlabel("frame #")
ylabel("line count")
printf("Line count: mean %d, std %d\n", mean(line_count(challenge)), std(line_count(challenge)))
printf("Wrong canny: mean %d, std %d\n", mean(wrong_canny_line_count(challenge)), std(wrong_canny_line_count(challenge)))

print(p, "illustrations/wrong_canny_line_count.png", '-dpng');

hold off;


##
## lane slope / intercept
##
p = figure();
hold on;

r = 4;
c = 2;

subplot(r, c, 1);
plot(left_lane_slope(challenge), 'r', right_lane_slope(challenge))
title("winner slope")
legend("left", "right")
subplot(r, c, 2);
plot(left_lane_intercept(challenge), 'r', right_lane_intercept(challenge))
title("winner intercept")
legend("left", "right")

subplot(r, c, 3);
plot(rgb_left_lane_slope(challenge), 'r', rgb_right_lane_slope(challenge))
title("rgb slope")
legend("left", "right")
subplot(r, c, 4);
plot(rgb_left_lane_intercept(challenge), 'r', rgb_right_lane_intercept(challenge))
title("rgb intercept")
legend("left", "right")

subplot(r, c, 5);
plot(wrong_canny_left_lane_slope(challenge), 'r', wrong_canny_right_lane_slope(challenge))
title("wrong canny slope")
legend("left", "right")
subplot(r, c, 6);
plot(wrong_canny_left_lane_intercept(challenge), 'r', wrong_canny_right_lane_intercept(challenge))
title("wrong canny intercept")
legend("left", "right")

subplot(r, c, 7);
plot(wrong_hough_left_lane_slope(challenge), 'r', wrong_hough_right_lane_slope(challenge))
title("wrong hough slope")
legend("left", "right")
subplot(r, c, 8);
plot(wrong_hough_left_lane_intercept(challenge), 'r', wrong_hough_right_lane_intercept(challenge))
title("wrong hough intercept")
legend("left", "right")

print(p, "illustrations/slope_intercept.png", "-dpng");

hold off;


##
## How do Hough affects slope and intercept
##
p = figure();
hold on;

r = 2;
c = 1;

subplot(r, c, 1);
plot(left_lane_slope(challenge), 'r', wrong_hough_left_lane_slope(challenge), 'b',
     right_lane_slope(challenge), 'r', wrong_hough_right_lane_slope(challenge), 'b')
title("Lane slope: Winner vs Wrong Hough")
legend("winner", "wrong hough")
xlabel("frame #")
ylabel("slope")

subplot(r, c, 2);
plot(left_lane_intercept(challenge), 'r', wrong_hough_left_lane_intercept(challenge), 'b',
     right_lane_intercept(challenge), 'r', wrong_hough_right_lane_intercept(challenge), 'b')
title("Lane intercept: Winner vs Wrong Hough")
legend("winner", "wrong hough")
xlabel("frame #")
ylabel("intercept")

print(p, "illustrations/wrong_hough.png", '-dpng');

hold off;

##
## Derivative for left lane slope
##
d1_left_lane_slope = diff(left_lane_slope(challenge));
d1_wrong_canny_left_lane_slope = diff(wrong_hough_left_lane_slope(challenge));

p = figure();
hold on;

plot(d1_left_lane_slope, 'r', d1_wrong_canny_left_lane_slope);
title("Left Lane: Winner Slope vs Wrong Hough Slope (Derivative 1)")
xlabel("frame #")
ylabel("slope diff")
legend("winner slope", "wrong hough slope")

print(p, "illustrations/d1_left_lane_slope.png", "-dpng");

hold off;
