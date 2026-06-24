import copy
import matplotlib.pylab as plt
import numpy as np
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split  # 训练集，测试集划分函数
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# import math
from math import cos, sin, sqrt,pi,atan2,tanh
from numba import jit
# This is a sample Python script.
import time
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
path_list=[]
step=0.2
# OBS=[[3,3]]
OBS=[[2,1],[3.0,4]]
point_record=[]
step_test=1
delta_theta=pi/4

@jit(nopython=True)
def point2line(x,y):
    A=1
    B=-1
    C=0
    d=abs((A*x+B*y+C))/(A**2+B**2)**0.5
    return d

@jit(nopython=True)
def Distance2d(x1,y1,x2,y2):
    return sqrt((x1-x2)**2+(y1-y2)**2)

# @jit(nopython=True)
# def Convolution(weights, features,count):
#     features2=[0 for i in range(18)]
#     for i in range(18):
#         features2[i] = sum([weights[i][j] * features[j][-1] for j in range(18)])
#     for i in range(18):
#         features[i][-1] = features2[i] / count[i]
#     return features

class Car():
    x=float()
    y=float()
    step=float()
    obs=[]
    def __init__(self,step,obs,x,y):
        self.x=x
        self.y=y
        self.x_=x
        self.y_=y
        self.step=step
        self.obs=copy.deepcopy(obs)
        self.obs_width=0.35
        self.dmin=0.05
        self.dmax=0.5
        self.target=[5,5]
        self.x_list=[]
        self.y_list=[]
        self.dire_list=[]
        self.obs_distance_list=[]
        self.obs_dire_list=[]
        self.target_distance_list=[]
        self.target_dire_list=[]
        self.features=[[] for i in range(18)]
        self.divi_dire=3
        self.d_mar=1.77
        self.convol=[[0 for i in range(18)] for j in range(18)]
        for i in range(18):
            for j in range(18):
                if abs(i-j)<9:
                    self.convol[i][j]=cos((j-i)/18*pi)
        self.count = [8, 9, 10, 11, 12, 13, 14, 15, 16, 16, 15, 14, 13, 12, 11, 10, 9, 8]
        self.features2=[0 for i in range(18)]

    # @jit(nopython=True)
    def move_0(self,direction):
        self.dire_list.append(direction)

        self.x += self.step * cos(direction)
        self.y += self.step * sin(direction)

        self.x_list.append(self.x)
        self.y_list.append(self.y)
    def move(self,direction):
        # time1=time.time()
        # self.x_=self.x
        # self.y_=self.y

        self.dire_list.append(direction)

        self.x += self.step * cos(direction)
        self.y += self.step * sin(direction)

        # target_dis=sqrt((self.x-self.target[0])**2+(self.y-self.target[1])**2)
        target_dis=Distance2d(self.x,self.y,self.target[0],self.target[1])
        idx_i=-1
        # obs_target = math.sqrt((self.target[0] - self.obs[idx_i][0]) ** 2 + (self.target[1] - self.obs[idx_i][1]) ** 2)
        # obs_dis = math.sqrt((self.x - self.obs[idx_i][0]) ** 2 + (self.y - self.obs[idx_i][1]) ** 2)
        for i in range(len(self.obs)):
            # obs_target=sqrt((self.target[0]-self.obs[i][0])**2+(self.target[1]-self.obs[i][1])**2)
            # obs_dis=sqrt((self.x-self.obs[i][0])**2+(self.y-self.obs[i][1])**2)
            obs_target=Distance2d(self.target[0],self.target[1],self.obs[i][0],self.obs[i][1])
            obs_dis=Distance2d(self.x,self.y,self.obs[i][0],self.obs[i][1])
            if obs_dis**2+target_dis**2>obs_target**2:
                idx_i=i
                break

        # self.obs_distance_list.append(obs_dis)
        # self.obs_dire_list.append(math.atan2(self.obs[idx_i][1]-self.y,self.obs[idx_i][0]-self.x))
        # self.target_distance_list.append(target_dis)
        # self.target_dire_list.append(math.atan2(self.target[1]-self.y,self.target[0]-self.x))
        # print(1111,time.time()-time1)
        # Fat_list = [0 for i in range(18)]

        features_local=[]

        uT = atan2(self.target[1] - self.y, self.target[0] - self.x)
        for i in range(18):
            x_e = self.x + cos(pi * i / 18-delta_theta) * self.step*step_test
            y_e = self.y + sin(pi * i / 18-delta_theta) * self.step*step_test
            c_p_sum = 0
            for j in range(len(self.obs)):
                # d_p = sqrt((x_e - self.obs[j][0]) ** 2 + (y_e - self.obs[j][1]) ** 2) - 0.5 - self.obs_width
                d_p = Distance2d(x_e,y_e,self.obs[j][0],self.obs[j][1]) - 0.5 - self.obs_width
                if d_p <= self.dmin:
                    c_p = 10
                elif d_p <= self.dmax:
                    c_p = (1 + tanh(1 / (d_p - self.dmin) + 1 / (d_p - self.dmax))) / 2
                else:
                    c_p = 0
                c_p_sum += c_p
            FaT = 1 / (1 + cos(i / 18 * pi-delta_theta - uT)) + c_p_sum*1+0.8*point2line(x_e,y_e)/self.d_mar
            # Fat_list[i] = FaT
            # self.features[i].append(FaT)
            features_local.append(FaT)

        # features2=[0 for i in range(18)]
        # count = [0 for i in range(18)]

        # print(2222, time.time() - time1)
        # for i in range(18):
        #     features2[i]=0
        #     for j in range(18):
        #         # if abs(i-j)<9:
        #         features2[i]+=self.convol[i][j]*self.features[j][-1]
        #             # count[i]+=1
        #     self.features[i][-1]=features2[i]/count[i]


        for i in range(18):
            # self.features2[i]=sum([self.convol[i][j]*self.features[j][-1] for j in range(18)])
            self.features2[i]=sum([self.convol[i][j]*features_local[j] for j in range(18)])

        # print(self.features2)
        for i in range(18):
            # self.features[i][-1]=self.features2[i]/self.count[i]
            self.features[i].append(self.features2[i]/self.count[i])

        # self.features=Convolution(self.convol,self.features,self.count)


        self.x_list.append(self.x)
        self.y_list.append(self.y)

    def PathPlan(self):
        FaTmin=1000
        Fat_list=[0 for i in range(18)]
        for i in range(18):
            x_e=self.x+cos(pi*i/18-delta_theta)*self.step*step_test
            y_e=self.y+sin(pi*i/18-delta_theta)*self.step*step_test
            c_p_sum=0
            for j in range(len(self.obs)):
                d_p=sqrt((x_e-self.obs[j][0])**2+(y_e-self.obs[j][1])**2)-0.5-self.obs_width
                if d_p<=self.dmin:
                    c_p=10
                elif d_p<=self.dmax:
                    c_p=(1+tanh(1/(d_p-self.dmin)+1/(d_p-self.dmax)))/2
                else:
                    c_p=0
                c_p_sum+=c_p
            uT=atan2(self.target[1]-self.y,self.target[0]-self.x)
            FaT=1/(1+cos(i/18*pi-delta_theta-uT))+c_p_sum*1+0.8*point2line(x_e,y_e)/self.d_mar
            Fat_list[i]=FaT
            if FaT<FaTmin:
                FaTmin=FaT
                index_min=i
        dire_min=index_min/18*pi-delta_theta
        # print(FaTmin,self.x,self.y,dire_min)
        FaTmin2=10000
        FaTmin3=10000
        for i in range(18):
            if abs(i-index_min)>=self.divi_dire and Fat_list[i]<FaTmin2:
                FaTmin2=Fat_list[i]
                index_min2=i
        for i in range(18):
            if abs(i-index_min)>=self.divi_dire and abs(i-index_min2)>=self.divi_dire and Fat_list[i]<FaTmin3:
                if FaTmin2!=i:
                    FaTmin3=Fat_list[i]
                    index_min3=i
        dire_min2=index_min2/18*pi-delta_theta
        dire_min3=index_min3/18*pi-delta_theta
        return dire_min,dire_min2,dire_min3


    def run(self):
        dire,_,_=self.PathPlan()
        self.move(dire)
        if Distance2d(self.x,self.y,self.target[0],self.target[1])<0.4:
            plt.scatter([self.obs[0][0]], [self.obs[0][1]],edgecolors='r')
            plt.plot(self.x_list, self.y_list)
            plt.plot()
            # show出图形
            plt.show()

def angle_calculate(x,y):
    arc1=atan2(x[1],x[0])
    arc2=atan2(y[1],y[0])
    arcx=arc1-arc2
    while(arcx>pi):
        arcx-=pi*2
    while(arcx<-pi):
        arcx+=pi*2
    if arcx<0:
        arcx=pi*2-arcx
    if arcx<pi/2:
        return 1
    else:
        return 0


def TreeSearch(x,y, path):
    global path_list
    global step
    global OBS,point_record
    if point_record!=[]:
        for i in range(len(point_record)):
            if Distance2d(x,y,point_record[i][0],point_record[i][1])<0.02:#--------------------
                return 0
            # else:
    point_record.append([x,y])
    # print(path)
    car = Car(step, OBS, x, y)
    path_e = copy.deepcopy(path)
    ut = atan2(car.target[1] - y, car.target[0] - x)
    dire=[0,0,0]
    if Distance2d(x, y, car.target[0], car.target[1]) < 0.3:
        path_list.append(path_e)
        return 1
    for i in range(len(OBS)):
        if Distance2d(x, y, car.obs[i][0], car.obs[i][1]) - 0.5-0.3 < 0.2:
            return 0
    flag=0
    for i in range(len(OBS)):
        if Distance2d(x, y, car.obs[i][0], car.obs[i][1]) < 2:
            flag=1
    if flag==0:
        path_e.append(ut)  # ut
        car.move(ut)
        result=TreeSearch(car.x,car.y,path_e)
        return result
        # if result==0:
        #     return 0,0
        # if result==1:
        #     return 1,path_e
    flag=0
    for i in range(len(OBS)):
        if Distance2d(car.obs[i][0], car.obs[i][1], car.target[0], car.target[1]) < \
                Distance2d(car.target[0], car.target[1], x, y):
            flag=1
    if flag==0:
        path_e.append(ut)  # ut
        car.move(ut)
        result=TreeSearch(car.x,car.y,path_e)
        return result
    # else:
    dire[0], dire[1], dire[2] = car.PathPlan()
    for i in range(3):
        car = Car(step, OBS, x, y)
        path_e = copy.deepcopy(path)
        path_e.append(dire[i])
        car.move(dire[i])
        result=TreeSearch(car.x,car.y,path_e)
    return result


class Net(nn.Module):
    def __init__(self,n_input,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden1 = nn.Linear(n_input,n_hidden)
        self.hidden2 = nn.Linear(n_hidden,n_hidden)
        self.predict = nn.Linear(n_hidden,n_output)
    def forward(self,input):
        out = self.hidden1(input)
        out = F.relu(out)
        out = self.hidden2(out)
        out = F.relu(out)
        out =self.predict(out)

        return out

if __name__ == '__main__':
    # car=Car(0.4,[[3,3]],0,0)
    # print(car.obs)
    # while(1):
    #     car.run()
    # global path_list
    start_t=time.time()
    TreeSearch(0,0,[])
    search_time=time.time()-start_t
    print("Tree Search Time:",search_time,"s")

    # print(path_list)
    plt.scatter([OBS[i][0] for i in range(len(OBS))], [OBS[i][1] for i in range(len(OBS))], edgecolors='r')
    plt.scatter([5], [5], edgecolors='g')
    for i in range(len(path_list)):
        car=Car(step, OBS, 0,0)
        for j in range(len(path_list[i])):
            car.move(path_list[i][j])

        plt.plot(car.x_list, car.y_list,linewidth =1)
        plt.plot()
    # show出图形
    # plt.show()
    len_min=10000
    index_minx=0
    for i in range(len(path_list)):
        if len(path_list[i])<len_min:
            len_min=len(path_list[i])
            index_minx=i

    car = Car(step, OBS, 0, 0)
    for j in range(len(path_list[index_minx])):
        car.move(path_list[index_minx][j])
    plt.plot(car.x_list, car.y_list,linewidth =4.0, color="green")
    # # plt.plot()
    # plt.grid()
    # # show出图形
    # plt.show()
    points=[]
    print("Tree Path Length:", step * len(car.x_list))
    for j in range(len(car.x_list)):
        for i in range(len(OBS)):
            if Distance2d(OBS[i][0],OBS[i][1],car.x_list[j],car.y_list[j])<0.2+0.9:
                points.append(car.x_list[j])
                continue
    print("Length Path:",len(points))



    net = Net(18, 50, 1)
    print(net)

    x=torch.randint(0, 5, size=(len(car.dire_list), 18), dtype=torch.float32)
    y=torch.tensor([[0] for i in range(len(car.dire_list))],dtype=torch.float32)
    for i in range(len(car.x_list)):
        for j in range(18):
            x[i][j]=car.features[j][i]
        # x[i][0]=car.obs_dire_list[i]
        # x[i][1]=car.obs_distance_list[i]
        # x[i][2]=car.target_dire_list[i]
        # x[i][3]=car.target_distance_list[i]

    for i in range(len(car.dire_list)):
        y[i][0]=(car.dire_list[i]+delta_theta)/(pi)
    print(x)
    print(y)
    x=Variable(x)
    y=Variable(y)

    print("number:", len(car.dire_list))
    print("Train_output:", car.dire_list)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
    loss_func = torch.nn.MSELoss()
    loss_list = []

    # for t in range(5000):
    count=0
    while(count<20000):
        count+=1
        prediction = net(x)
        loss = loss_func(prediction, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(count,loss.data)
        loss_list.append(loss.data)
        if loss.data<0.004:
            break
    print(prediction-y)
    print(prediction)
    print(y)
    #测试
    car = Car(step, OBS, 0, 0)
    direction=pi/4
    start_t = time.time()
    # plt.ion()

    x = torch.randint(0, 5, size=(1, 18), dtype=torch.float32)
    y = torch.tensor([[0]], dtype=torch.float32)
    while(1):


        flag = 0
        for i in range(len(OBS)):
            if Distance2d(car.obs[i][0], car.obs[i][1], car.target[0], car.target[1]) < \
                    Distance2d(car.target[0], car.target[1], car.x, car.y):
                flag = 1
        if flag == 0:

            direction = atan2(car.target[1] - car.y, car.target[0] - car.x)
            car.move_0(direction)
        else:
            car.move(direction)
            # x[0][0] = car.obs_dire_list[-1]
            # x[0][1] = car.obs_distance_list[-1]
            # x[0][2] = car.target_dire_list[-1]
            # x[0][3] = car.target_distance_list[-1]
            for j in range(18):
                x[0][j]=car.features[j][-1]
            # print(x[0])
            y[0][0] = car.dire_list[-1]
            x = Variable(x)
            # y = Variable(y)
            prediction = net(x)
            direction=(prediction[0][0])*(pi)-delta_theta

        # car.dire_list.append(direction)
        # print(flag,prediction[0][0],direction)

        # plt.cla()
        #
        # plt.plot(car.x_list, car.y_list, linewidth=4.0, color="red")
        # plt.plot()
        # plt.grid()
        # # show出图形
        # plt.show()
        # plt.pause(0.1)
        print(flag)
        if Distance2d(car.x, car.y, car.target[0], car.target[1]) < 0.2:
            break
    print("Tree Search Time:", search_time, "s")
    print("GCN Search Time:",time.time()-start_t,"s")
    print("GCN Path Length:",step*len(car.x_list))
    # plt.cla()
    print("Output:",car.dire_list)
    plt.plot(car.x_list, car.y_list, linewidth=4.0, color="red")
    plt.plot()
    plt.grid()
    # show出图形
    plt.show()
    # plt.pause(10000)
    plt.plot([i for i in range(len(loss_list))],loss_list, color="red")
    plt.grid()
    plt.show()

