import numpy as np
import os, psutil
import sys
from sklearn import svm
from sklearn.metrics import confusion_matrix
from joblib import dump, load
from sklearn.model_selection import cross_val_score
import io
from collections import deque
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.model_selection import GridSearchCV
import xml.etree.ElementTree as ET

def conv_1d(data_array, conv_size):
    if conv_size != 1:
        new_database = np.ones((1,18001))
        for i in range(0,data_array.shape[0]):
            listOfData = []
            stack = deque()
            for j in range(0,conv_size):
                stack.append(0)
            listOfData.append(round(sum(list(stack))/len(list(stack)),2))
            for k in range(1,data_array.shape[1]):
                stack.popleft()
                if k+conv_size >= data_array.shape[1]:
                    stack.append(0.0)
                else:
                    stack.append(data_array[i,k+int((conv_size-1)/2)])
                listOfData.append(round(sum(list(stack))/len(list(stack)),4))
            new_database = np.vstack((new_database, np.array(listOfData, dtype = float)))
            stack.clear()
            
        new_database = np.delete(new_database, 0, axis=0)
        return new_database
    else:
        return data_array

def add_to_data_array(data, mass_data, intensity_data,diap_mz):
        
        if data is None:
                if diap_mz == 0:
                        data = np.zeros(5001)
                elif diap_mz == 1:
                        data = np.zeros(18001)
        else:
                if diap_mz == 0:
                        data = np.vstack((data, np.zeros(5001)))
                        for i in range(len(mass_data)):
                                data[-1,mass_data[i]] = intensity_data[i]
                elif diap_mz == 1:
                        data = np.vstack( (data, np.zeros(18001)) )
                        for i in range(len(mass_data)):
                                data[-1,mass_data[i]-2000] = intensity_data[i]
        return data

def xmlFileRead(file_path):
        parser = ET.XMLParser(encoding="utf-8")
        tree = ET.parse(file_path)
        root = tree.getroot()
        mass_list = []
        intensity_list = []
        for child in root:
            mass_list.append(int(float(child.find('mass').text)))
            intensity_list.append(float(child.find('absi').text))
        return mass_list, intensity_list

def search_xml(ppath):
    for q in os.listdir(path=ppath):
        if (q == 'peaklist.xml'):
            xml_file_path.append(os.path.join(ppath, q))
        elif (os.path.isdir(os.path.join(ppath,q))):
            search_xml(os.path.join(ppath, q))


#print('Input path to spectrum data')
#path_to_data = input()
path_to_data = 'D:/Machine Learning for MALDI-TOF spectrum/MALDI-spectrum_november 2022/26_08_2022/High mass_pos ions_after bl subtract'
#go to folder with spectrum data
os.chdir(path_to_data)
d_mz = 1
#create data array
data = add_to_data_array(None,[],[],d_mz)
xml_file_path = []


#search peaklists in folder-tree 
for name_of_folder in os.listdir(path='.'):
    if os.path.isdir('./'+name_of_folder):
        ppath = os.path.join('.',name_of_folder)
        search_xml(ppath)
        

class_label = np.zeros((len(xml_file_path)))
label = []

for i in range(len(xml_file_path)):
        ppath = xml_file_path[i]
        if ppath.split('\\')[2] not in label:
                label.append(ppath.split('\\')[2])
                if i > 0:
                        class_label[i] = class_label[i-1]+1
        else:
                class_label[i] = class_label[i-1]
        m,i = xmlFileRead(ppath)
        data = add_to_data_array(data, m, i, d_mz)

X = np.delete(data, 0, axis=0)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=42, stratify = Y)
#SVM + 1Dconv
for size in range(1,15,2):
    X_been = conv_1d(X_train,size)
    X_been_n = normalize(X_been, axis=1, norm='l2')
    parameters = {'kernel':['rbf'], 'C':np.logspace(-5, 5, 11), 'gamma':np.logspace(-4, 5, 10)}
    classificator = svm.SVC(decision_function_shape = 'ovo')
    clf = GridSearchCV(classificator, parameters, cv = 4)
    clf.fit(X_been_n,y_train)
    print(size, clf.cv_results_)


