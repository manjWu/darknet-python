import cv2, time
import tools
import multiprocessing
import numpy as np
import sharedmem

'''
使用帧间差分+SR对图片进行处理
processVideo(result_list, filePath, diff_threshhold)

'''

# 6/10 多进程不能减少处理时间，能显示图片，可能是multiprocessing.manager共享数据导致
# 6/11 改进MG共享数据导致的速度问题，但是图片无法显示
# Process finished with exit code 139 (interrupted by signal 11: SIGSEGV)（将显示程序imshow放在主程序可显示）
# 6/12 将sharedmem.empty(shape[0:2], dtype)放在主程序中（所以子程序必须在同一文件，才会有image_in变量）


filePath = '/Volumes/MachsionHD/drone_实验视频/实验原始视频（部分转换格式）/20190403紫金港/20190403紫金港450m-实验2/20190403_20190403144034_20190403145700_144034-全景.mp4'
threshhold_diff = 25 # 帧间差分阈值
threshhold_SR = 30
flag_multiprocess = 0
flag_diff = 0
flag_SR = 1
gain = 0.3 #显示缩放比例


##############################目标检测函数#################################################
def processVideo(result_list, filePath, diff_threshhold):


    startframe = 5
    frame_num = startframe
    endframe = 18000  # int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap = cv2.VideoCapture(filePath)

    while(cap.isOpened()):


        print("frame_num:%d " % frame_num)

        ############################## 0 read frames ##############################################
        # 若是第一帧
        if frame_num == startframe:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, pre_BGR_frame = cap.read()  # read image
            # 降采样、转化灰度图、高斯滤波
            # pre_BGR_frame = getDownsamplingImg(pre_BGR_frame)
            pre_frame = cv2.cvtColor(pre_BGR_frame, cv2.COLOR_BGR2GRAY)  # convert RGB -> gray
            # pre_frame = pre_frame[0: 1200, :]  # delete the ground backgroud



        # 若非第一帧
        if frame_num > startframe and frame_num <= endframe:
            ret, cur_BGR_frame = cap.read()
            cur_frame = cv2.cvtColor(cur_BGR_frame, cv2.COLOR_BGR2GRAY)  # convert RGB -> gray

            if flag_multiprocess ==0:
                if flag_diff == 1:
                    # 帧间差分
                    start = time.time()
                    diffMap, cnts_diff = tools.frame_diff(cur_frame, pre_frame, diff_threshhold)
                    diff_BGR_frame = tools.drawRectangle(cnts_diff, cur_BGR_frame, (0, 0, 255))
                    # 显示
                    cv2.imshow('Normalized Interframe Difference Map', tools.getDownsamplingImg(diffMap, gain))
                    # cv2.imshow("Original Frame", tools.getDownsamplingImg(diff_BGR_frame,gain))

                if flag_SR == 1:
                # 谱残差
                    saliencyMap, cnts_SR = tools.SR(cur_frame, threshhold_SR)
                    SR_BGR_frame = tools.drawRectangle(cnts_SR, cur_BGR_frame, (0, 0, 255))
                    cv2.imshow('Saliency Map', tools.getDownsamplingImg(saliencyMap, gain))
                    cv2.imshow("Original Frame", tools.getDownsamplingImg(SR_BGR_frame, gain))


                # press keyboard 'q' to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            else:
                # # multiprocess很慢，能显示图片
                # time1 = time.time()
                # result_list[1], cnts_diff = tools.frame_diff(cur_frame, pre_frame, diff_threshhold)
                # result_list[2] = tools.drawRectangle(cnts_diff, cur_BGR_frame, (0, 0, 255))
                # result_list[3] = tools.SR(pre_frame)
                # result_list[0] = 1
                # time2 = time.time()
                # print("process time: ", time2 - time1)

                # # multiprocess faster不能用子程序不显示
                # count = 1
                # start = time.time()
                # for image in diffMap,diff_BGR_frame,saliencyMap:
                #     image_in[:] = image.copy()
                #     result_list[count] = image_in
                #     count += 1
                # end = time.time()
                # print("copy time:", end-start)
                # result_list[0] = 1

                # multiprocess faster可用
                time1 = time.time()
                image_in1[:], cnts_diff = tools.frame_diff(cur_frame, pre_frame, threshhold_diff)
                image_in2[:] = tools.drawRectangle(cnts_diff, cur_BGR_frame, (0, 0, 255))
                image_in3[:], cnts_diff = tools.SR(cur_frame, threshhold_SR)
                count = 1
                for image_in in image_in1,image_in2,image_in3:
                    result_list[count] = image_in
                    count += 1
                result_list[0] = 1
                time2 = time.time()
                print("process time: ", time2 - time1)

            # 更新背景
            pre_frame = cur_frame
        frame_num += 1

    print("video ending...")
    cap.release()
    cv2.destroyAllWindows()



def refreshShow(result_list,gain):
    while True:
        if result_list[0] == 1:
            print("refresh begin...")
            time3 = time.time()
            cv2.imshow('Normalized Interframe Difference Map', tools.getDownsamplingImg(result_list[1], gain))
            cv2.imshow("Original Frame", tools.getDownsamplingImg(result_list[2], gain))
            cv2.imshow('Saliency Map', tools.getDownsamplingImg(result_list[3], gain))
            # press keyboard 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            result_list[0] = 0
            time4 = time.time()
            print("print time: ", time4 - time3)


if __name__ == "__main__":

    if flag_multiprocess ==0:
        # 单进程
        processVideo([0, 0, 0, 0], filePath, threshhold_diff)
    else:
        # 多进程
        cap = cv2.VideoCapture(filePath)
        with multiprocessing.Manager() as MG:  # 重命名#
            result_list = MG.list()
            result_list.append(0)
            for i in range(3):
                result_list.append([])

            _, image = cap.read()
            shape = np.shape(image)
            dtype = image.dtype
            # 此处必须对应三种输出的shape
            image_in1 = sharedmem.empty(shape[0:2], dtype)
            tup = (3,)
            image_in2 = sharedmem.empty(shape[0:2]+tup, dtype)
            image_in3 = sharedmem.empty(shape[0:2], "float64")

            p1_img = multiprocessing.Process(target=processVideo, args=(result_list, filePath, threshhold_diff))  # 创建新进程1
            p2_show = multiprocessing.Process(target=refreshShow, args=(result_list,gain))  # 创建新进程2
            p1_img.start()
            p2_show.start()
            # 当处理程序结束后终止显示程序
            p1_img.join()
            # p2_show进程里是死循环，无法等待其结束，只能强行终止
            p2_show.terminate()
            print('Child process end.')