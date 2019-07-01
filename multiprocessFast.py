import numpy as np
import cv2, multiprocessing, sharedmem

'''
示例：sharedmem
'''

def show_image(images):

    cap = cv2.VideoCapture("/Users/wumanjia/Desktop/test.mp4")
    assert cap.isOpened(), 'Cannot capture source'

    while 1:

        ret, image = cap.read()
        if not ret:
            break
        image_in[:] = image.copy()
        images[0] = image_in
        cv2.imshow("avi",image_in)
        cv2.waitKey(1)

def aa(images):
    while 1:
        print("aa")

        cv2.imshow("a",images[0])
        cv2.waitKey(1)

if __name__ == '__main__':
    # cap = cv2.VideoCapture('1.avi')
    # 若放在子程序中子程序会报错（aa），且若子程序与主程序不在同一文件也会报错
    # cap = cv2.VideoCapture("/Volumes/MachsionHD/drone_实验视频/实验原始视频（部分转换格式）/20190403紫金港/20190403紫金港450m-实验2/20190403_20190403144034_20190403145700_144034-全景.mp4")

    # 必须放在主程序中建立
    image_in = sharedmem.empty((1536, 2048, 3),"uint8")

    images = multiprocessing.Manager().list()
    images.append(0)

    a = multiprocessing.Process(target=show_image,args=(images,))
    a.start()

    multiprocessing.Process(target=aa, args=(images,)).start()
    a.join()