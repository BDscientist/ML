import os
import sys
import io
import urllib.request as req
import requests,json
import urllib.parse as rep
from bs4 import BeautifulSoup


sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')




input_x = [1,2,3,4,5,6,7,8,9,10]
inpuy_y = [10,11,12,13,14,15,16,17,18,19]


class regression:
    def __init__(self,x=[],y=[]):
        self.x=x
        self.y=y
        self.meanx=0
        self.meany=0

    def mean_x(self):
        meanx = 0
        sum =0
        for i in self.x:
            sum +=i
        meanx = sum /len(self.x)
        return meanx

    def mean_y(self):
        meany = 0
        sum =0
        for i in self.y:
            sum +=i
        meany = sum /len(self.y)
        return meany

    def gradient_b1(self):
        mean_x1 = self.meanx
        #mean_x()
        #new_meanx = mean_x1.meanx

        mean_y1 = self.meany
        #mean_y()
        #new_meany = mean_y1.meany

        b1=[]
        b2=[]

        new_sum =0
        new_sum2=0
        result=0

        for i in self.x:
            b1.append((( mean_x1 - self.x[i] ) *( mean_y1 - self.y[i] )))
            new_sum2 += ((self.x[i] - mean_x1)**2)

        for j in b1:
            new_sum +=j

        result = new_sum / new_sum2
        return result

    def gradient_b0(self):
        b0=0
        bb = gradient_b1()
        f_b1 = bb.result

        new_meanx2 = mean_x()
        f_meanx = new_meanx2.meanx

        new_meany2 = mean_y()
        f_meany = new_meanx2.meany

        b0 = f_meany - (f_b1 * f_mean)

        return b0



test  = regression([1,2,3,4,5,6,7,8,9,10],[10,11,12,13,14,15,16,17,18,19])

print(test.x)
print(test.y)
print(test.mean_x())
print(test.gradient_b1())
