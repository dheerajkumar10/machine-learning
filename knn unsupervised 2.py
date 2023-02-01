#!/usr/bin/env python
# coding: utf-8

# In[122]:


import numpy as np
import math
import pandas as pd


# In[123]:


def euclidean_distance(row1,row2):
    distance=0.0;
    for i in range(len(row1)-1):
        distance = distance + (row1[i] - row2[i])**2
        sqrtdistance = math.sqrt(distance)
    return sqrtdistance


# In[124]:


dataset1 = [(( 1.6398767752769, 88.883711494291, 31), 'M' ),
(( 1.6793587237148, 78.251133790593, 33), 'M' ),
(( 1.5606515359382, 80.379998232509, 31), 'M' ),
(( 1.7041530344833, 79.042241386106, 32), 'W' ),
(( 1.7902744499504, 85.490626529836, 29), 'M' ),
(( 1.9114216491429, 88.583502900854, 32), 'M' ),
(( 1.663548694822, 79.334614233063, 40), 'W' ),
(( 1.617632971184, 75.775365798322, 31), 'M' ),
(( 1.7860510851332, 79.669328833931, 30), 'M' ),
(( 1.7471626838103, 91.554297596111, 37), 'M' ),
(( 1.6556706330438, 75.342441750817, 35), 'W' ),
(( 1.6286051561942, 82.313561191238, 35), 'W' ),
(( 1.9686795347355, 88.412367429298, 28), 'W' ),
(( 1.6536849541587, 81.876114419651, 37), 'M' ),
(( 1.7610802870024, 82.988874559137, 32), 'M' ),
(( 1.6532023744521, 75.112316142275, 36), 'W' ),
(( 1.6488888245147, 85.514239601957, 33), 'M' ),
(( 1.8999672054599, 88.901713931905, 28), 'M' ),
(( 1.966309884977, 85.960854309352, 34), 'W' ),
(( 1.7419989152201, 76.660687308549, 34), 'W' ),
(( 1.7410759420504, 88.45516564909, 28), 'W' ),
(( 1.6128717285128, 81.665533303814, 33), 'W' ),
(( 1.6564895965606, 75.620689385906, 37), 'W' ),
(( 1.7211014604543, 82.167763986892, 30), 'M' ),
(( 1.7561740337006, 83.307181550011, 33), 'M' ),
(( 1.829358792818, 79.556738736726, 31), 'M' ),
(( 1.6988479546535, 72.10698614958, 36), 'W' ),
(( 1.7001898133107, 77.667564587703, 30), 'W' ),
(( 1.8318014740353, 79.401723669333, 34), 'W' ),
(( 1.7099296818001, 79.91983280517, 33), 'M' ),
(( 1.7122213159364, 80.630692309157, 27), 'W' ),
(( 1.8055173032312, 82.277319781475, 35), 'W' ),
(( 1.7139583336454, 84.392207033842, 32), 'W' ),
(( 1.6497621909206, 86.700815495777, 32), 'W' ),
(( 1.7587413893208, 81.862466664499, 28), 'M' ),
(( 1.7920179991514, 83.054454585374, 29), 'M' ),
(( 1.7185356648902, 88.04269294477, 30), 'M' ),
(( 1.7299525968726, 84.863853163678, 32), 'W' ),
(( 1.7128072110792, 75.247967717682, 28), 'M' ),
(( 1.7090113354922, 81.975751013121, 38), 'M' ),
(( 1.6408347829316, 85.532149194174, 36), 'M' ),
(( 1.9439955398178, 91.884484957824, 28), 'W' ),
(( 1.7274177034241, 90.680829821807, 29), 'M' ),
(( 1.7431383422182, 79.396931399357, 29), 'M' ),
(( 1.7627304528441, 77.823612566684, 34), 'M' ),
(( 1.697511194971, 83.219306060898, 36), 'M' ),
(( 1.7144352542511, 74.447483201619, 30), 'W' ),
(( 1.9320520994348, 87.364788215837, 33), 'M' ),
(( 1.7848924060926, 90.275129770064, 36), 'M' ),
(( 1.8158317123786, 79.843909017857, 34), 'W' ),
(( 1.6504722768288, 79.584402934576, 31), 'W' ),
(( 1.7927896572691, 87.693456960208, 39), 'M' ),
(( 1.6788550699291, 75.452096416904, 30), 'W' ),
(( 1.6906902585759, 90.721382227158, 32), 'M' ),
(( 1.7969231045757, 81.175470577644, 32), 'W' ),
(( 1.8272198495309, 78.281776425164, 29), 'W' ),
(( 1.5780872658714, 73.663400417089, 34), 'W' ),
(( 1.8541763770393, 82.572365028261, 28), 'M' ),
(( 1.7572046635299, 80.51849073263, 35), 'M' ),
(( 1.7186152115652, 86.220888855088, 33), 'W' ),
(( 1.6641907978382, 79.210050597573, 26), 'W' ),
(( 1.5199771169132, 71.317841016816, 33), 'W' ),
(( 1.6250655934722, 74.407003024961, 32), 'W' ),
(( 1.8675473239669, 84.448244601362, 29), 'W' ),
(( 1.8404486486111, 89.426699375902, 29), 'W' ),
(( 1.8757743059545, 81.999002746239, 35), 'M' ),
(( 1.7632815092034, 79.36824086749, 35), 'M' ),
(( 1.8132125109131, 82.643741757309, 26), 'W' ),
(( 1.7547850607383, 84.541706225928, 32), 'W' ),
(( 1.7703652978655, 79.83725077508, 32), 'W' ),
(( 1.7580369855044, 86.597354438986, 26), 'M' ),
(( 1.7540620222022, 70.509003009041, 32), 'W' ),
(( 1.6943549127417, 79.566630883349, 28), 'M' ),
(( 1.7406163378013, 85.112612000104, 30), 'W' ),
(( 1.7363444470048, 88.908813091598, 34), 'M' ),
(( 1.7657736823754, 89.65008339188, 35), 'M' ),
(( 1.7000682489752, 82.319801914129, 31), 'M' ),
(( 1.8287655119064, 82.096196760201, 30), 'W' ),
(( 1.6012802654675, 78.36689506328, 32), 'W' ),
(( 1.5567397767717, 76.97634718814, 36), 'W' ),
(( 1.6359947298447, 74.084357364286, 43), 'W' ),
(( 1.8480482716516, 89.617957957534, 29), 'M' ),
(( 1.7950508480979, 85.134689958236, 32), 'M' ),
(( 1.7741176279525, 81.226426984581, 37), 'M' ),
(( 1.5184339504451, 75.479618720447, 35), 'W' ),
(( 1.8098540158159, 79.544455119207, 35), 'W' ),
(( 1.8775407864433, 95.542290746789, 38), 'M' ),
(( 1.666959379114, 77.424974269731, 32), 'M' ),
(( 1.8160944808652, 88.0756292322, 35), 'M' ),
(( 1.7852503682952, 80.025155244015, 36), 'W'),
(( 1.7997278803457, 84.730546785786, 29), 'M' ),
(( 1.8553132021196, 85.372906688875, 39), 'M' ),
(( 1.7402195975081, 80.755452542812, 36), 'M' ),
(( 1.6935735594697, 90.300889406346, 35), 'M' ),
(( 1.6479845599911, 82.279738450111, 32), 'M' ),
(( 1.7497769481617, 81.854049161695, 36), 'W' ),
(( 1.6131867795702, 79.679699684502, 28), 'W' ),
(( 1.6785597961667, 79.605139914867, 27), 'W' ),
(( 1.7990235331177, 81.326076437711, 35), 'W' ),
(( 1.8004516454367, 87.334640212661, 36), 'M' ),
(( 1.7752079108488, 75.374505132426, 33), 'W' ),
(( 1.7788378081896, 95.569026528677, 31), 'M' ),
(( 1.6332949719903, 67.084064485731, 39), 'W' ),
(( 1.7470876651889, 84.060086797003, 32), 'W' ),
(( 1.6903417883801, 88.157990527444, 36), 'M' ),
(( 1.5755320781775, 81.339292392716, 36), 'W' ),
(( 1.8613846838884, 80.617099812841, 34), 'W' ),
(( 1.7255361874771, 79.04831015098, 39), 'W' ),
(( 1.721946923349, 76.237725758393, 33), 'W' ),
(( 1.8764388000423, 87.201679825193, 33), 'M' ),
(( 1.8643130511338, 84.441050707272, 34), 'M' ),
(( 1.6795079349624, 77.485796908393, 33), 'W' ),
(( 1.7861997754645, 85.122321137222, 30), 'M' ),
(( 1.7835146320575, 80.885455045068, 31), 'M' ),
(( 1.7492347000938, 91.940494960301, 30), 'M' ),
(( 1.7346249569595, 80.461035920457, 34), 'M' ),
(( 1.7475710365745, 82.417748592106, 36), 'M' ),
(( 1.6369916904163, 63.694409193906, 34), 'W' ),
(( 1.7279105168181, 78.127793305004, 32), 'W' ),
(( 1.8962250581481, 85.784793130339, 32), 'W' )]

trainDataset = dataset1[:20]
testDataset = dataset1[-100:]
testGender = list(a[-1] for a in testDataset)
testData = list(a[0] for a in testDataset)

trainGender = list(a[-1] for a in trainDataset)
trainData = list(a[0] for a in trainDataset)


# In[125]:


def get_neighbours(train,test_row,k):

    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row,train_row)
        distances.append((train_row,dist))
    distances.sort(key=lambda tup:tup[1])
    neighbours = list()

    for i in range(k):
        neighbours.append(distances[i][0])
    return neighbours


# In[126]:


correctIndex = list()
wrongIndex = list()
def accuracy(testData,testGender,trainData,trainGender):
    count=0
    correctIndex=[]
    wrongIndex=[]
    for i in range(len(testData)):
        genderM=0
        genderW=0
        testData1= testData.pop(i)
        test_neighbours = get_neighbours(trainData,testData1,3)
        testData.insert(i,testData1)
        for neighbour in test_neighbours:
            y = trainData.index(neighbour)
            if(trainGender[y] == 'M'):
                genderM = genderM+1
            else:
                genderW = genderW+1
        if(genderM >genderW):
            accuracyGender = 'M'
        else:
            accuracyGender = 'W'
    
        if(accuracyGender ==testGender[i] ):
            count = count+1
            correctIndex.append(i)
        else:
            wrongIndex.append(i)
    
    accuracyPred = count/len(testData)
    print(accuracyPred)
    return correctIndex


# In[127]:


cIndex=list(accuracy(testData,testGender,trainData,trainGender))


# In[128]:


trainGenderAll = trainGender.copy()
trainDataAll = trainData.copy()
testGenderAll = testGender.copy()
testDataAll = testData.copy()
for i in range(len(cIndex)):
    trainDataAll.append(testData[cIndex[i]])
    trainGenderAll.append(testGender[cIndex[i]])
    testDataAll.remove(testData[cIndex[i]])
    testGenderAll.remove(testGender[cIndex[i]])
accuracy(testDataAll,testGenderAll,trainDataAll,trainGenderAll)


# In[129]:


trainGender_1 = trainGender.copy()
trainData_1 = trainData.copy()
testGender_1 = testGender.copy()
testData_1 = testData.copy()
for i in range(len(cIndex)):
    trainData_1.append(testData[cIndex[i]])
    trainGender_1.append(testGender[cIndex[i]])
    testData_1.remove(testData[cIndex[i]])
    testGender_1.remove(testGender[cIndex[i]])
    accuracy(testData_1,testGender_1,trainData_1,trainGender_1)


# In[130]:


trainGender_5 = trainGender.copy()
trainData_5 = trainData.copy()
testGender_5 = testGender.copy()
testData_5 = testData.copy()
c5=0
for i in range(len(cIndex)):
    trainData_5.append(testData[cIndex[i]])
    trainGender_5.append(testGender[cIndex[i]])
    testData_5.remove(testData[cIndex[i]])
    testGender_5.remove(testGender[cIndex[i]])
    c5=c5+1
    if(c5==5):
        accuracy(testData_5,testGender_5,trainData_5,trainGender_5)
        c5=0


# In[131]:


trainGender_10 = trainGender.copy()
trainData_10 = trainData.copy()
testGender_10 = testGender.copy()
testData_10 = testData.copy()
c10=0
for i in range(len(cIndex)):
    trainData_10.append(testData[cIndex[i]])
    trainGender_10.append(testGender[cIndex[i]])
    testData_10.remove(testData[cIndex[i]])
    testGender_10.remove(testGender[cIndex[i]])
    c10=c10+1
    if(c10==10):
        accuracy(testData_10,testGender_10,trainData_10,trainGender_10)
        c10=0


# In[ ]:


#from the above obtained we can see that the accuracy percentage for KNN applied on 100 test data from 20 trained data is 64%
#When we added untrained labels with correct prediction as labelled data in one go and and tested for failed testcases only 
# with 35 failed cases as test cases and 85 trainedsamples our accuracy for that 35 test cases is 38% but when we changed our
# adding into trained list with different number of values being added for each iterartion our accuracy for prediction
#we are taking max predcition for each accuracy level we got in iterations so for 1 point being added in iterativerly we achieved
# max accuracy as 67% for failed test cases
# when we add 5 points to trained data iteratively our accuracy level dropped to 65% and when add 10 points to trained data interatively
# our accuracy is dropped to 64% in predicting the test cases.


# with this if we keep on adding failed points in the test and success test cases in training we can achieve all unlabelled 
#data as labelled data after certain number of iterations.

