import cv2
import numpy as np
from scipy.interpolate import RectBivariateSpline
from external_energy import external_energy
from internal_energy_matrix import get_matrix

global x_max, y_max

def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        #save point
        raw_xs.append(x)
        raw_ys.append(y)

        #display point
        cv2.circle(img_copy, (x, y), 3, (128, 128, 0), -1) # thickness = -1 fills the circle with color
        cv2.imshow('image', img_copy)

def clamp(pts, v_max, v_min):
    pts[pts > v_max] = v_max
    pts[pts < v_min] = v_min
    return pts

# this function is bugged, use scipy.interpolate package instead
def bilinear_interpolation(xs, ys, f):
    x1 = np.floor(xs).astype(int)
    x2 = np.ceil(xs).astype(int)
    y1 = np.floor(ys).astype(int)
    y2 = np.ceil(ys).astype(int)
    x1 = clamp(x1, x_max, 0)
    x2 = clamp(x2, x_max, 0)
    y1 = clamp(y1, y_max, 0)
    y2 = clamp(y2, y_max, 0)

    int_x = x1 == x2 
    int_y = y1 == y2
    if int_x.any() or int_y.any(): print(xs[int_x], ys[int_y]) 

    # try:        
    f_at_x_y1 = (x2-xs)/(x2-x1+1e-6) * f[y1, x1] + ((xs-x1)/(x2-x1+1e-6) * f[y1, x2])
    f_at_x_y2 = (x2-xs)/(x2-x1+1e-6) * f[y2, x1] + ((xs-x1)/(x2-x1+1e-6) * f[y2, x2])
    if int_x.any():
        f_at_x_y1[int_x] = f[y1, xs.astype(int)[int_x]]
        f_at_x_y2[int_x] = f[y2, xs.astype(int)[int_x]]
    f_at_x_y = (y2-ys)/(y2-y1+1e-6) * f_at_x_y1 + ((ys-y1)/(y2-y1+1e-6) * f_at_x_y2)
    if int_y.any():
        f_at_x_y[int_y] = f_at_x_y1[int_y] # f_at_x_y1 == f_at_x_y2
    # except IndexError as e:
        # print(e)
        # print("x1, y1, x2, y2: ", x1.max(), x1.min(), y1.max(), y1.min(), x2.max(), x2.min(), y2.max(), y2.min())
    # f_at_x_y[np.isnan(f_at_x_y)] = f[x1[np.isnan(f_at_x_y)], y1[np.isnan(f_at_x_y)]]
    return f_at_x_y

if __name__ == '__main__':
    #point initialization
    # img_path = 'images/brain.png'
    # img_path = 'images/circle.jpg'
    # img_path = 'images/dental.png'
    # img_path = 'images/shape.png'
    # img_path = 'images/square.jpg'
    # img_path = 'images/star.png'
    img_path = 'images/vase.tif'
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_copy = img.copy()
    x_max = len(img[0]) - 1 # x is col num
    y_max = len(img) - 1 # y is row num

    raw_xs = []
    raw_ys = []
    cv2.imshow('image', img_copy)

    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #selected points are in xs and ys
    #interpolate between the selected points
    in_xs = []
    in_ys = []
    n = 50
    for i in range(len(raw_xs)):
        if raw_xs[i] < (raw_xs[i+1] if i<len(raw_xs)-1 else raw_xs[0]):
            steps = np.linspace(raw_xs[i], (raw_xs[i+1] if i<len(raw_xs)-1 else raw_xs[0]), n) # steps is monotonously increasing
            flipped = False
        else:
            steps = np.linspace((raw_xs[i+1] if i<len(raw_xs)-1 else raw_xs[0]), raw_xs[i], n)
            flipped = True

        y_endpts = raw_ys[i:][:2] if i<len(raw_xs)-1 else [raw_ys[i], raw_ys[0]]
        interps = np.interp(steps, [steps[0],steps[-1]], y_endpts if not flipped else y_endpts[::-1])

        in_xs.append(steps[:-1]) if not flipped else in_xs.append(steps[::-1][:-1])
        in_ys.append(interps[:-1]) if not flipped else in_ys.append(interps[::-1][:-1])

    in_xs = np.round(np.ravel(in_xs)).astype(int)
    in_xs = clamp(in_xs, x_max, 0)
    in_ys = np.round(np.ravel(in_ys)).astype(int)
    in_ys = clamp(in_ys, y_max, 0)

    alpha = 0.001
    beta = 20
    gamma = 0.5
    kappa = 1
    num_points = len(in_xs)

    #get matrix
    M = get_matrix(alpha, beta, gamma, num_points)

    img_blur = cv2.GaussianBlur(img, (5,5), 0)
    #get external energy
    w_line = 1
    w_edge = 2
    w_term = 2
    E = external_energy(img_blur, w_line, w_edge, w_term)

    intp = RectBivariateSpline(np.arange(E.shape[1]), np.arange(E.shape[0]), E.T, kx=2, ky=2, s=0)
    # fx = cv2.Sobel(E, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
    # fy = cv2.Sobel(E, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)

    iter = 0
    xs = np.copy(in_xs)
    ys = np.copy(in_ys)
    #optimization loop
    while True:
        # fx_at_x_y = fx[ys, xs] if iter == 0 else bilinear_interpolation(xs, ys, fx) 
        # fy_at_x_y = fy[ys, xs] if iter == 0 else bilinear_interpolation(xs, ys, fy)
        fx_at_x_y = intp(xs, ys, dx=1, grid=False)
        fy_at_x_y = intp(xs, ys, dy=1, grid=False)
        xs_next = M @ (gamma*xs - (kappa*fx_at_x_y))
        ys_next = M @ (gamma*ys - (kappa*fy_at_x_y))
        iter += 1

        termination_value = (np.mean(np.abs(xs_next - xs)), np.mean(np.abs(ys_next - ys)))
        print(termination_value)
        if (np.mean(np.abs(xs_next - xs)) < 5e-3 and np.mean(np.abs(ys_next - ys)) < 5e-3) or iter > 1e4:
        # if iter > 1e4:
            break
        else:
            xs = xs_next
            ys = ys_next
        img_canvas = img.copy()
        img_canvas = cv2.cvtColor(img_canvas, cv2.COLOR_GRAY2BGR)
        contour = np.stack(((np.round(xs)).astype(int), (np.round(ys)).astype(int)), axis=1)[:, np.newaxis, :]
        cv2.drawContours(img_canvas, contour, -1, (128, 128, 0), 3)
        cv2.imshow('image', img_canvas)
    # print("Xs: ", xs, "/nYs: ", ys)
    print("Iterations: ", iter)

    # contour = np.stack(((np.round(xs)).astype(int), (np.round(ys)).astype(int)), axis=1)[:, np.newaxis, :]
    # cv2.drawContours(img_canvas, contour, -1, (128, 128, 0), 3)

    # filename = 'brain_result.jpg'
    # filename = 'circle_result.jpg'
    # filename = 'dental_result.jpg'
    # filename = 'shape_result.jpg'
    # filename = 'square_result1.jpg'
    # filename = 'star_result.jpg'
    filename = 'vase_result.jpg'
    cv2.imwrite(filename, img_canvas)

    cv2.imshow('image', img_canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()