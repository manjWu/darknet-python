# coding=utf-8
import cv2
import numpy as np
import tools
from sklearn.cluster import KMeans

threshold_min=15
threshold_max=20
search_step=5
show_gain = 0.3
# img_4 = np.concatenate((img,boundary),axis = 2)

def Calculate_border(dst,gard_threshhold_t):
    boundary_tmp = np.zeros(shape=dst.shape[1])
    for col in list(range(dst.shape[1])):
        for row in list(range(dst.shape[0])):
            if dst[row][col] > gard_threshhold_t:
                boundary_tmp[col] = row
                break
    return boundary_tmp

def get_vector3d_mean(img,boundary_tmp):
    num_sky = 0
    num_ground = 0
    mean_sky = np.zeros(shape=(3,))
    mean_ground = np.zeros(shape=(3,))

    for col in range(img.shape[1]):
        for row in range(img.shape[0]):
            if row<boundary_tmp[col]:
                num_sky += 1
                mean_sky += img[row][col]
            else:
                num_ground +=1
                mean_ground += img[row][col]
    mean_sky /= num_sky
    mean_ground /= num_ground

    return mean_sky,mean_ground,num_sky,num_ground

def get_Jnt(img,boundary_tmp,mean_sky,mean_ground,num_sky,num_ground):
    lamada = 2
    sky_covariance_matrix = np.zeros(shape=(3,3))
    ground_covariance_matrix = np.zeros(shape=(3,3))
    for col in range(img.shape[1]):
        for row in range(img.shape[0]):
            if row < boundary_tmp[col]:
                tmp = img[row][col]-mean_sky
                sky_covariance_matrix += tmp*tmp.T
            else:
                tmp = img[row][col]-mean_ground
                ground_covariance_matrix += tmp*tmp.T
    sky_covariance_matrix /= num_sky
    ground_covariance_matrix /= num_ground
    sky_eigen_value, sky_eigen_vector = np.linalg.eig(sky_covariance_matrix)
    ground_eigen_value, ground_eigen_vector = np.linalg.eig(sky_covariance_matrix)
    sky_det = np.linalg.det(sky_covariance_matrix)
    ground_det = np.linalg.det(ground_covariance_matrix)
    Jnt = 1/(lamada*sky_det+ground_det+lamada*max(np.abs(sky_eigen_value))+max(np.abs(ground_eigen_value)))
    return Jnt

def get_sky_img(img, bopt):
    for col in range(img.shape[1]):
        for row in range(img.shape[0]):
            if row > bopt[col]:
                img[row][col] = 0
    return(img)
#==============get_img_with_sky_opt======================
def absolute_border_dif(bopt):
    tmp = 0
    num = len(bopt)
    for i in range(num-1):
        tmp += abs(bopt[i]-bopt[i+1])
    return tmp/num

def get_threshold(img):
    H=img.shape[0]
    thresh = [0,0,0,0]
    thresh[0]=H/20 # thresh_avg_min
    thresh[1]=H/10 # thresh_avg_large
    thresh[2]=15 # thresh_absolute_border_dif
    thresh[3]=H/3 # thresh4
    return thresh

def diff_abs(bopt,thresh):
    tmp = 0
    num = len(bopt)
    check = 0
    for i in range(num):
        if abs(bopt[i]-bopt[i+1])>thresh[3]:
            check = 1
            break
    return check

def get_original_skypixel_num(bopt):
    tmp = 0
    for i in range(len(bopt)):
        tmp+=bopt[i]
    return tmp

def Euclidean_distance(input1,input2):
    return np.sqrt(np.power(input1 - input2, 2).sum())

def low_textrue_check(img,SizeN):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    check_lowtextrue = img.copy()
    row = int(SizeN/2)
    while(row<img.shape[0]-SizeN/2):
        col = int(SizeN/2)
        while(col<img.shape[1]-SizeN/2):
            tmp = 0
            i = -int(SizeN/2)
            while(i<=SizeN/2):
                j = -int(SizeN/2)
                while (j <= SizeN / 2):
                    # print(j)
                    tmp += abs(img[row+i,col+j]-img[row,col])
                    j += 1
                i += 1
            tmp = tmp/(SizeN*SizeN)
            if tmp<2.5:
                check_lowtextrue[row][col] = 0
            col+=1
        row += 1

    return check_lowtextrue

def classify_sky_byK_means(bopt,img):
    clusterCount = 2
    totle_sky_num = get_original_skypixel_num(bopt)
    # print("total_pixel:",img.shape[0]*img.shape[1])
    # print("total_sky_pixel:",totle_sky_num)
    X = []
    G = []
    s_row = []
    s_col = []
    final_out = np.zeros(shape=img.shape[0:3])
    for col in list(range(img.shape[1])):
        for row in list(range(img.shape[0])):
            if row < bopt[col]:
                X.append(img[row][col])
                s_col.append(col) #把sky列号存起来
                s_row.append(row)
                final_out[row,col] = 255  # 将sky部分置为255
            else:
                G.append(img[row][col])
    y_pred = KMeans(n_clusters=clusterCount, random_state=0).fit_predict(X)
    X = np.array(X)
    G = np.array(G)
    s_col = np.array(s_col).reshape(X.shape[0],1)
    s_row = np.array(s_row).reshape(X.shape[0], 1)
    y_pred = y_pred.reshape(X.shape[0],1)
    result = np.concatenate((X,y_pred,s_row,s_col),axis = 1) #0-2列RGB三色，3列分类结果，4列对应的行数，
    ############## 对聚类结果进行处理
    # 计算s1，s2，G平均值
    s1_mean = np.mean(result[np.where(result[:,3]==0)][:,0:3],axis= 0)
    s2_mean = np.mean(result[np.where(result[:,3]==1)][:,0:3],axis= 0)
    s_mean = [s1_mean,s2_mean]
    G_mean = np.mean(G[:,0:3],axis= 0)
    # 计算s1和s2与地面的欧式距离
    s_ground_M = [Euclidean_distance(s1_mean,G_mean),Euclidean_distance(s2_mean,G_mean)]
    # 计算两类天空的列号方差
    sky1_col_variance = np.var(result[np.where(result[:, 3]==0)][5])
    sky2_col_variance = np.var(result[np.where(result[:, 3] == 1)][5])
    sky_variance_ratio = sky1_col_variance / sky2_col_variance; # 用列号方差约束边缘白墙

    '''第一关还未复现'''
    # 判别当前帧的天空区域：第一关
    # 在视频流上，先验知识：
    # 1 前后帧的天空 区域大小不会突变
    # 2 前后帧的天空 区域均值也不会突变
    # 先根据这两条 确定聚类后的天空区域哪一类是天空
    # 如果有一类区域既满足平均灰度值与以前帧真实天空灰度值相近，也满足区域大小（像素个数）与以前帧真实天空区域大小相近，则认为此类区域是当前帧的天空区域
    '''第二/三关'''
    ## 判别当前帧的天空区域：第二关
    # 实际天空一般是很集中区域， 经聚类后的两类区域， 我们认为如果有一类区域
    # 列号方差太大，也就是此类区域很松散 ，就认为此类区域是伪天空区域。

    ##  判别当前帧的天空区域：第三关
    # 如果两类区域列号方差相差不是太大，则用这两类区域分别于ground的距离
    # 做真实天空区域判定,即 距离大的为真实天空区域。
    if sky_variance_ratio>1.4:
        s_true = 1
    elif sky_variance_ratio<0.7:
        s_true = 0
    else:
        if s_ground_M[0]>s_ground_M[1]:
            s_true = 0
        else:
            s_true = 1
    s_true_sky = result[np.where(result[:, 3] == s_true)]
    sky_opt_mean = np.mean(s_true_sky[:,0:3],axis= 0)

    '''往下寻找天空'''
    check_lowtextrue = low_textrue_check(img, 3)
    cv2.imshow("lowtextrue",check_lowtextrue)
    # cv2.waitKey(0)
    bopt_opt = np.zeros(shape=bopt.shape)


    for i in range(len(s_true_sky)-1):
        if s_true_sky[i+1][5]!=s_true_sky[i][5]: # 得到分界线
            bopt_opt[s_true_sky[i][5]]=s_true_sky[i][4]

    cv2.imshow("s_true",final_out)

    for col in list(range(img.shape[1])):
        for row in list(range(img.shape[0])):
            if row > bopt_opt[col]:
                distance = Euclidean_distance(img[row][col],sky_opt_mean)
                if distance<100 and check_lowtextrue[row][col]==0:
                    final_out[row][col] = 255

    return final_out

#========================================================

def get_img_with_sky(img,threshold_min, threshold_max, search_step):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 获得sobel边缘图像
    print("1. Sobel begin...")
    x = cv2.Sobel(img_gray, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img_gray, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)  # 转回uint8
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    print("2. 根据能量函数自适应调节边界阈值...")
    n = (threshold_max - threshold_min) / search_step + 1
    Jmax = 0
    for k in range(int(n)):
        grad_threahold_t = threshold_min+(threshold_max-threshold_min)/(n-1)*k
        print("   gard_threahold_t:",grad_threahold_t)
        # 计算天空边缘分界线
        # print("Calculate_border begin...")
        boundary_tmp = Calculate_border(dst,grad_threahold_t)
        # print("get_vector3d_mean begin...")
        mean_sky,mean_ground,num_sky,num_ground = get_vector3d_mean(img,boundary_tmp)
        # print("get_Jnt begin...")
        Jtmp = get_Jnt(img,boundary_tmp,mean_sky,mean_ground,num_sky,num_ground)
        if Jtmp>Jmax:
            print("   更新Jnt...")
            Jmax = Jtmp
            bopt = boundary_tmp.copy()
            gradd = grad_threahold_t
        else:
            break
    print("   扫描后的阈值: ",gradd)
    return bopt


def get_img_with_sky_opt(img, threshold_min, threshold_max, search_step):
    # Detection of the Image without a Sky Region
    bopt = get_img_with_sky(img, threshold_min, threshold_max, search_step)
    border_Avg = np.mean(bopt)
    abs_border_dif = absolute_border_dif(bopt)
    thresh = get_threshold(img)
    cv2.imshow("未优化前",get_sky_img(img, bopt))
    if (diff_abs(bopt,thresh)):
        print("3. 边界有突变，使用Kmeans对天空区域进行聚类，再重新扫描")
        final_out = classify_sky_byK_means(bopt, img)
        return final_out

    else:
        print("3. 边界无突变")
        img_sky = get_sky_img(img, bopt)  # 得到最后的边界，显示
        return img_sky



if __name__ == "__main__":
    img = cv2.imread("/Users/wumanjia/Desktop/test2.png")
    img = tools.getDownsamplingImg(img,0.4)
    img_sky = get_img_with_sky_opt(img,threshold_min, threshold_max, search_step)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

