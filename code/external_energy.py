import cv2
import numpy as np

def line_energy(image):
    #implement line energy (i.e. image intensity)
    i = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    return i

def edge_energy(image):
    #implement edge energy (i.e. gradient magnitude)
    # Sobel kernal (TODO:try Scharr kernal)
    gx = cv2.Sobel(image, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3) # CV_64F(float) does not work
    gy = cv2.Sobel(image, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
    g = np.sqrt((gx)**2 + ((gy)**2))
    # g = np.abs(gx.astype(float)) + np.abs(gy.astype(float))
    g = cv2.normalize(g, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    return np.negative(g)

def term_energy(image):
    #implement term energy (i.e. curvature)
    #reduce noise
    cx = cv2.Sobel(image, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
    cy = cv2.Sobel(image, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)

    cxx = cv2.Sobel(cx, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
    cxy = cv2.Sobel(cx, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
    cyy = cv2.Sobel(cy, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)

    c = (cxx * (cy**2)) - (2 * cxy * cx * cy) + (cyy * (cx**2))
    divisor = ((cx**2 + cy**2)**1.5)
    # print("divisor: ", divisor, "MIN: ", np.min(divisor), np.isnan(divisor).sum())
    c = cv2.divide(c, divisor)
    c[np.isnan(c)] = 0
    c = cv2.normalize(c, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    return c

def external_energy(image, w_line, w_edge, w_term):
    #implement external energy
    e_line = line_energy(image)
    e_edge = edge_energy(image)
    # print("Edge:", e_edge)
    e_term = term_energy(image)
    return w_line * e_line + w_edge * e_edge + w_term * e_term
