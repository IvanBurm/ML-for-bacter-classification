import os
import sys
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import numpy as np
from sklearn.decomposition import PCA

def add_to_data_array(data, mass_data, intensity_data):
        if data is None:
                data = np.zeros(18001)
        else:
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
           
print("START PROGRAMM")
data = add_to_data_array(None,[],[])
xml_file_path = []
#go to folder with spectrum data
os.chdir('D:/Machine Learning for MALDI-TOF spectrum/MALDI-spectrum_november 2022/26_08_2022/High mass_pos ions_after bl subtract')
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
        data = add_to_data_array(data, m, i)

data = np.delete(data, 0, axis=0)
print(data.shape)

data = data.reshape(-1,715)
print(data.shape)
n_comp = data.shape[0]
pca = PCA(n_components = data.shape[0])
pca.fit(data)
explained = pca.explained_variance_ratio_
print(pca.components_)
plt.figure(figsize=(10, 6))
plt.scatter(np.arange(1, n_comp + 1), explained)
plt.plot(np.arange(1, n_comp + 1), explained)
plt.title('Dependence of  variance on the number of components',size=14)
plt.xlabel('Num of components', size=14)
plt.ylabel('proportion of the explained variance', size=14)

plt.xlim(0,1000)
plt.show()

print("END PROGRAMM")
