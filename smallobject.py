import os
import re

# 提取数据文件夹的各个子文件夹的txt文件中
# 


if __name__ == '__main__':

	path = '/Volumes/MachsionHD/drone_dataset3_08.10'
	savepath = '/Users/wumanjia/c++/darknet/drone/'
	txtname = "smaller"
	threshold = 0.01

def str2float(str):
	strcut = str.split(".",-1)
	float = int(strcut[0])+int(strcut[1])/(10**len(strcut[1]))
	return float



def smallextract():
	f = open(savepath+txtname+".txt", "w")
	dirlist = os.listdir(path)
	num = 0
	for dir in dirlist: #遍历文件夹，得到子文件夹
		dirpath = os.path.join(path,dir); #组合文件路径以及文件名
		if os.path.isdir(dirpath):
			filelist = os.listdir(dirpath)
			for filelabel in filelist: #遍历子文件夹
				filetype = os.path.splitext(filelabel)[1] #文件扩展名
				if filetype in ['.jpeg','.png','.jpg']:
					filename = os.path.splitext(filelabel)[0]
					label = open(dirpath+'/'+filename+'.txt').readlines()
					for labelcell in label:
						labelcut = re.split(' ',labelcell)
						# labelcell.split(" ",-1)#str=" ", num=-1
						if (str2float(labelcut[3])<=threshold) & (str2float(labelcut[4])<=threshold): #长宽分别小于40 30
							num += 1
							new_context = '/root/wmj-docker/droneDataset/'+dir+'/'+filelabel+'\n'
							f.write(new_context)
							print(num)
							print(new_context)


smallextract()
