import cv2
import numpy as np

np.set_printoptions(suppress=True)
# Setting matching points in first image
xy_1 = np.array([[157, 32],  # x1[0][0], y1[0][1]
                 [211, 37],  # x2[1][0], y2[1][1]
                 [222, 107],  # x3[2][0], y3[2][1]
                 [147, 124]]).astype('float32')  # x4[3][0], y4[3][1]

# Setting matching points in second image
xy_2 = np.array([[6, 38],  # x'1[0][0], y'1[0][1]
                 [56, 31],  # x'2[1][0], y'2[1][1]
                 [82, 85],  # x'3[2][0], y'3[2][1]
                 [22, 118]]).astype('float32')  # x'4[3][0], y'4[3][1]


arrayA = np.array([[xy_1[0][0], xy_1[0][1], 1, 0, 0, 0, -xy_1[0][0] * xy_2[0][0], -xy_1[0][1] * xy_2[0][0],-xy_2[0][0]],
                   [0, 0, 0, xy_1[0][0], xy_1[0][1], 1, -xy_1[0][0] * xy_2[0][1], -xy_1[0][1] * xy_2[0][1],-xy_2[0][1]],
                   [xy_1[1][0], xy_1[1][1], 1, 0, 0, 0, -xy_1[1][0] * xy_2[1][0], -xy_1[1][1] * xy_2[1][0],-xy_2[1][0]],
                   [0, 0, 0, xy_1[1][0], xy_1[1][1], 1, -xy_1[1][0] * xy_2[1][1], -xy_1[1][1] * xy_2[1][1],-xy_2[1][1]],
                   [xy_1[2][0], xy_1[2][1], 1, 0, 0, 0, -xy_1[2][0] * xy_2[2][0], -xy_1[2][1] * xy_2[2][0],-xy_2[2][0]],
                   [0, 0, 0, xy_1[2][0], xy_1[2][1], 1, -xy_1[2][0] * xy_2[2][1], -xy_1[2][1] * xy_2[2][1],-xy_2[2][1]],
                   [xy_1[3][0], xy_1[3][1], 1, 0, 0, 0, -xy_1[3][0] * xy_2[3][0], -xy_1[3][1] * xy_2[3][0],-xy_2[3][0]],
                   [0, 0, 0, xy_1[3][0], xy_1[3][1], 1, -xy_1[3][0] * xy_2[3][1], -xy_1[3][1] * xy_2[3][1],-xy_2[3][1]]])


def getPerspectiveTransformMatrix(p1, p2):
    arrayA = np.array(
        [[p1[0][0], p1[0][1], 1, 0, 0, 0, -p1[0][0] * p2[0][0], -p1[0][1] * p2[0][0], -p2[0][0]],
         [0, 0, 0, p1[0][0], p1[0][1], 1, -p1[0][0] * p2[0][1], -p1[0][1] * p2[0][1], -p2[0][1]],
         [p1[1][0], p1[1][1], 1, 0, 0, 0, -p1[1][0] * p2[1][0], -p1[1][1] * p2[1][0], -p2[1][0]],
         [0, 0, 0, p1[1][0], p1[1][1], 1, -p1[1][0] * p2[1][1], -p1[1][1] * p2[1][1], -p2[1][1]],
         [p1[2][0], p1[2][1], 1, 0, 0, 0, -p1[2][0] * p2[2][0], -p1[2][1] * p2[2][0], -p2[2][0]],
         [0, 0, 0, p1[2][0], p1[2][1], 1, -p1[2][0] * p2[2][1], -p1[2][1] * p2[2][1], -p2[2][1]],
         [p1[3][0], p1[3][1], 1, 0, 0, 0, -p1[3][0] * p2[3][0], -p1[3][1] * p2[3][0], -p2[3][0]],
         [0, 0, 0, p1[3][0], p1[3][1], 1, -p1[3][0] * p2[3][1], -p1[3][1] * p2[3][1], -p2[3][1]]])

    print(arrayA.shape)
    U, S, Vh = np.linalg.svd(arrayA)
    print(Vh,'==========================')
    print(Vh[-1, :],'-----------------')

    print(Vh[-1, -1],'+++++++++++++++++++++++')

    L = Vh[-1, :] / Vh[-1, -1]
    H = L.reshape(3, 3)
    return H



a=cv2.getPerspectiveTransform(xy_1,xy_2)
b=getPerspectiveTransformMatrix(xy_1,xy_2)
print('')
print(arrayA)
print('')
print(a)
print('''''')
print(b)




































