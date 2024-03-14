import cv2 as cv
import numpy as np
import os

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope, intercept = parameters
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    if np.isnan(np.sum(left_fit_average)) or np.isnan(np.sum(right_fit_average)):
        return None
    if left_fit_average[0] == 0 or right_fit_average[0] == 0:
        return None
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])

def thres(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    blur = cv.GaussianBlur(gray,(5, 5), 0)
    # canny = cv.Canny(blur, 50, 150) # detect edges
    canny = cv.threshold(blur, 130, 145, cv.THRESH_BINARY)[1]
    return canny

def display_lines(image, lines):
    height, width = image.shape[:2]
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            if (x1 >= 0 and x1 < width) and (x2 >= 0 and x2 < width):
                cv.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 3)
        if len(lines) == 2:
            # squares = np.array([[(lines[0][0], lines[0][1]), (lines[0][2], lines[0][3]), (lines[1][2], lines[1][3]), (lines[1][0], lines[1][1])]])
            # S = cv.contourArea(squares[0])
            # if S > 28000 or S < 20000:
            #     return line_image
            arr = [(lines[0][0], lines[0][1]), (lines[0][2], lines[0][3]), (lines[1][2], lines[1][3]), (lines[1][0], lines[1][1])]        
            cv.fillPoly(line_image, pts=np.array([arr]), color=(0, 100, 0))
    return line_image

def region_of_interest(image):
    height, width = image.shape[:2]
    polygons = np.array([[
        (0, height),
        (width, height),
        (325, 145)
    ]])
    mask = np.zeros_like(image)
    cv.fillPoly(mask, polygons, 255)
    masked_image = cv.bitwise_and(image, mask)
    return masked_image


frames = os.listdir('frames')
frames.sort(key=lambda x: int(x.split('.')[0]))
if not os.path.exists('output_frames'):
    os.mkdir('output_frames')
for i in range(len(frames)):
    filename = f'frames/{i}.png'
    frame = cv.imread(filename)
    thres_image = thres(frame)
    cropped_image = region_of_interest(thres_image)
    lines = cv.HoughLinesP(cropped_image, 1, np.pi/180, 30, maxLineGap=200)
    if lines is None:
        continue
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    if np.all(line_image == np.zeros_like(frame)):
        continue
    combo_image = cv.addWeighted(frame, 0.8, line_image, 1, 1)
    cv.imwrite(f'output_frames/{i}.png', combo_image)