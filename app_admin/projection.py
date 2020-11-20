import numpy as np
import cv2


def l1_distance(a, b):
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**(1/2)


def projection(img, pt_lu, pt_ru, pt_ld, pt_rd, window_height, window_width):
    """    
    投影変換
    """
    points_before_projection = np.float32(
        [pt_lu, pt_ru, pt_ld, pt_rd])
    points_after_projection = np.float32([[0.0, 0.0], [window_width, 0.0], [
        0.0, window_height], [window_width, window_height]])
    projection_matrix = cv2.getPerspectiveTransform(
        points_before_projection, points_after_projection)
    dst = cv2.warpPerspective(img, projection_matrix,
                              (window_width, window_height))
    return dst


img = cv2.imread("./cards.png")
print(img.shape)
cv2.imshow("img", img)


#　対応する頂点は固定
pt_lu = np.float32([190, 125])
pt_ru = np.float32([282, 91])
pt_ld = np.float32([278, 224])
pt_rd = np.float32([378, 182])


window_width = 400
window_height = int(window_width *
                    l1_distance(pt_lu, pt_ld) / l1_distance(pt_lu, pt_ru))


projection_img = projection(
    img, pt_lu, pt_ru, pt_ld, pt_rd, window_height, window_width)
cv2.imshow("projection img", projection_img)
cv2.waitKey()
