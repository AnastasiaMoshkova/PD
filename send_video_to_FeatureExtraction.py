import os
import subprocess
os.chdir('C:\\Users\\Asus\\Desktop\\OpenFace_2.2.0_win_x64')

def video_to_openface(path):
    listLM=[]
    listAU=[]
    for file in os.listdir(path):
        if 'mp4' in file:
            if (file.split('.')[1]=='mp4'):
                if (file.split('_')[0] in ['p2','p3','p5','p11']):
                    listAU.append(file)


    for i in range(len(listAU)):
       subprocess.run('FeatureExtraction.exe -f ' + path + listAU[i] + ' -out_dir ' + path  + ' -aus')

basepath = 'C:\\Users\\Asus\\Desktop\\Patient\\patient'
for i in ['44']:
    path = basepath+i+'\\'+ 'face'+'\\'
    video_to_openface(path)

