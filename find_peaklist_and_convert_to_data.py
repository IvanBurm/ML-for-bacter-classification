import os
import sys
import xml.etree.ElementTree as ET
import numpy as np

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


print('Input path to spectrum data')
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

print(class_label)
X = np.delete(data, 0, axis=0)
np.save('data',X)
np.save('class_label',class_label)
