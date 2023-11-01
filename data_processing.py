import os
import subprocess
import re
import math
import PSpincalc as sp
import numpy as np
import json
import cv2


def rename_face_files(path):
    os.mkdir(path + "face")
    listAU = []
    name = ['p1', 'p2', 'p3', 'p5', 'p11', 'p12_1', 'p12_2', 'p12_3', 'p12_4', 'p12_5', 'p12_6', 'p13_1', 'p13_2',
            'p13_3', 'p13_4', 'p13_5', 'p13_6']
    patient = path.split('\\')
    for file in os.listdir(path):
        if 'mp4' in file:
            if (file.split('.')[1] == 'mp4'):
                listAU.append(file)
    for i in range(len(listAU)):
        os.rename(path + listAU[i],
                  path + "face" + '\\' + name[i] + '_' + patient[-2] + '.mp4')


def video_to_FeatureExtraction(path, path_to_openface):
    path = path + 'face' + '\\'
    listAU = []
    os.chdir(path_to_openface)
    for file in os.listdir(path):
        if 'mp4' in file:
            if (file.split('.')[1] == 'mp4'):
                if (file.split('_')[0] in ['p2', 'p3', 'p5', 'p11']):
                    listAU.append(file)

    for i in range(len(listAU)):
        subprocess.run('FeatureExtraction.exe -f ' + path + listAU[i] + ' -out_dir ' + path + ' -aus')


def video_to_frames(video, path_output_dir, number):
    # extract frames from a video and save to directory as 'x.png' where
    # x is the frame index
    vidcap = cv2.VideoCapture(video)
    count = 0
    while vidcap.isOpened():
        success, image = vidcap.read()
        if success:
            if count % number == 0:
                print(os.path.join(path_output_dir, '%d.png') % count)
                cv2.imwrite(os.path.join(path_output_dir, '%d.png') % count, image)
                # cv2.imshow('picture',image)
                # cv2.waitKey(0)
            count += 1
        else:
            break
    cv2.destroyAllWindows()
    vidcap.release()


def faceImage(path):
    path = path + 'face' + '\\'
    files = os.listdir(path)
    for file in files:
        if 'mp4' in file:
            if file.split('_')[0] == 'p12':
                folder_frame = path + file.split('_')[0] + '_' + file.split('_')[1]
                os.mkdir(folder_frame)
                video_to_frames(path + file, folder_frame, 5)
                # print(folder_frame,file)
            if file.split('_')[0] == 'p13':
                folder_frame = path + file.split('_')[0] + '_' + file.split('_')[1]
                os.mkdir(folder_frame)
                video_to_frames(path + file, folder_frame, 5)
                # print(folder_frame,file)
            if file.split('_')[0] == 'p1':
                folder_frame = path + 'p1'
                os.mkdir(folder_frame)
                video_to_frames(path + file, folder_frame, 60)
                # print(folder_frame,file)


def video_to_FaceLandmarkImg(path, path_to_openface):
    os.chdir(path_to_openface)
    path = path + 'face' + '\\'
    task = ['p12_1', 'p12_2', 'p12_3', 'p12_4', 'p12_5', 'p12_6', 'p13_1', 'p13_2', 'p13_3', 'p13_4', 'p13_5',
            'p13_6', 'p1']
    for j in range(len(task)):
        subprocess.run('FaceLandmarkImg.exe -fdir ' + path + task[j] + ' -out_dir ' + path + task[j] + ' -aus')


def send_lmt_to_LM(path, path_to_RecordPlaybackSample):
    os.chdir(path_to_RecordPlaybackSample)
    os.mkdir(path + 'hand')
    patient = path.split('\\')
    for m in os.listdir(path):
        if m == 'm1' or m == 'm2':
            files = os.listdir(path + '\\' + m)
            for file in files:
                if file.split('.')[1] == 'lmt':
                    subprocess.run(
                        'RecordPlaybackSample.exe ' + path + '\\' + m + '\\' + file + " " + path  + "hand" + '\\' +
                        file.split('.')[0] + '_' + m + '_' + patient[-2] + '.txt')


def LMJson(path):
    path = path + 'hand'
    for the_file in os.listdir(path):
        if "txt" in the_file:
            print('file', the_file)
            f = open(os.path.join(path, the_file), 'r', encoding='utf-8', errors='ignore')

            dict_frame = {}
            dict_finger = {}
            d = {}
            d2 = {}
            res = []
            FINGER = ["THUMB_MCP", "THUMB_PIP", "THUMB_DIP", "THUMB_TIP",
                      "FORE_MCP", "FORE_PIP", "FORE_DIP", "FORE_TIP",
                      "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
                      "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
                      "LITTLE_MCP", "LITTLE_PIP", "LITTLE_DIP", "LITTLE_TIP"]
            for line in f:
                print(line)
                if line.find("frame") >= 0:
                    count = 0
                    a1 = line.lstrip(' ')
                    a2 = a1.rstrip(' ')
                    a3 = a2.split('.')
                    a4 = a3[0].split()

                    frame_number = int(a4[1])
                    # print(frame_number)
                    # print(dict_finger)

                if line.find("Hand id") >= 0:
                    count = 0
                    result = re.search(r'\((.*?)\)', line).group(1)
                    result = result.split()
                    print(result)
                    result_x = float(result[0].split(',')[0])
                    result_y = float(result[1].split(',')[0])
                    result_z = float(result[2].split(',')[0])
                    result_w = float(result[2].split(',')[1])
                    result_wx = float(result[3].split(',')[0])
                    result_wy = float(result[4].split(',')[0])
                    result_wz = float(result[5])
                    # print(result_x, " ",result_y, " ",result_z )
                    quaternion = np.array(
                        [result_w * (math.sqrt(result_wz ** 2 + result_wy ** 2 + result_wx ** 2 + result_w ** 2)),
                         result_wz, result_wy, result_wx])
                    ea = sp.Q2EA(quaternion, EulerOrder="zyx", ignoreAllChk=True)[0]
                    # res = (-ea[2] / (2 * 3.14)) * 360
                    if line.find("right hand") >= 0:
                        resCentre = (-ea[2] / (2 * 3.14)) * 360
                        if (resCentre < -150):
                            resCentre = 180
                        d2 = {'X': result_x, 'Y': result_y, 'Z': result_z, 'W': result_w, 'Wx': result_wx,
                              'Wy': result_wy,
                              'Wz': result_wz, 'Angle': resCentre}
                        d = {'CENTRE': d2}
                        hand = "right hand"
                        # dict_finger={'right hand':d}
                        # dict_finger['right hand']['CENTRE']['X']=result_x
                        # dict_finger['right hand']['CENTRE']['Y'] =result_y
                        # dict_finger['right hand']['CENTRE']['Z'] =result_z
                    if line.find("left hand") >= 0:
                        hand = "left hand"
                        resCentre = (ea[2] / (2 * 3.14)) * 360
                        if (resCentre < -150):
                            resCentre = 180
                        # dict_finger['left hand']['CENTRE']['X'] =result_x
                        # dict_finger['left hand']['CENTRE']['Y'] =result_y
                        # dict_finger['left hand']['CENTRE']['Z'] =result_z
                        d2 = {'X': result_x, 'Y': result_y, 'Z': result_z, 'W': result_w, 'Wx': result_wx,
                              'Wy': result_wy,
                              'Wz': result_wz, 'Angle': resCentre}
                        # print(d2)
                        d = {'CENTRE': d2}

                        # dict_finger = {'left hand': d}
                if line.find("bone with position") >= 0:
                    count = count + 1
                    # print(count)
                    result = re.search(r'\((.*?)\)', line).group(1)
                    result = result.split()
                    result_x1 = float(result[0].split(',')[0])
                    result_y1 = float(result[1].split(',')[0])
                    result_z1 = float(result[2].split(',')[0])
                    result_x2 = float(result[3].split(',')[0])
                    result_y2 = float(result[4].split(',')[0])
                    result_z2 = float(result[5].split(',')[0])
                    result_x3 = float(result[6].split(',')[0])
                    result_y3 = float(result[7].split(',')[0])
                    result_z3 = float(result[8].split(',')[0])
                    result_w3 = float(result[9])
                    # print(result_x1, " ", result_y1, " ", result_z1," ",result_x2, " ", result_y2, " ", result_z2," ",result_x3, " ", result_y3, " ", result_z3)
                    '''
                    if 'left hand' in dict_finger.keys():
                        hand="left hand"
                    else:
                        hand = "right hand"
                        '''
                    quaternionFinger = np.array(
                        [result_w3 * (math.sqrt(result_z3 ** 2 + result_y3 ** 2 + result_x3 ** 2 + result_w3 ** 2)),
                         result_z3, result_y3, result_x3])
                    eaF = sp.Q2EA(quaternionFinger, EulerOrder="zyx", ignoreAllChk=True)[0]
                    resFinger = (eaF[2] / (2 * 3.14)) * 360

                    d2 = {'X1': result_x1, 'Y1': result_y1, 'Z1': result_z1, 'X2': result_x2, 'Y2': result_y2,
                          'Z2': result_z2, 'X3': result_x3, 'Y3': result_y3, 'Z3': result_z3, 'W': result_w3,
                          'Angle': resFinger}
                    d.update({FINGER[count - 1]: d2})
                    dict_finger = {hand: d}
                    # dict_finger.update(d)
                    '''
                    dict_finger[hand][FINGER[count - 1]]['X1'] = result_x1
                    dict_finger[hand][FINGER[count - 1]]['X2'] = result_x2
                    dict_finger[hand][FINGER[count - 1]]['X3'] = result_x3

                    dict_finger[hand][FINGER[count - 1]]['Y1'] = result_y1
                    dict_finger[hand][FINGER[count - 1]]['Y2'] = result_y2
                    dict_finger[hand][FINGER[count - 1]]['Y3'] = result_y3

                    dict_finger[hand][FINGER[count - 1]]['Z1'] = result_z1
                    dict_finger[hand][FINGER[count - 1]]['Z2'] = result_z2
                    dict_finger[hand][FINGER[count - 1]]['Z3'] = result_z3

            '''
                    if (count == 20):
                        dict_finger.update({'frame': frame_number})
                        res.append(dict_finger)
                        dict_finger = {}
                        d = {}
                        d2 = {}

            pathSave = path
            with open(os.path.join(pathSave, the_file.split('.')[0] + '.json'),
                      'w') as outfile:
                json.dump(res, outfile)
            f.close()


basepath = 'C:\\Users\\Asus\\Desktop\\Parkinson\\Patient\\'
path_to_openface = 'C:\\Users\\Asus\\Desktop\\Parkinson\\OpenFace_2.2.0_win_x64'
path_to_RecordPlaybackSample = 'C:\\Users\\Asus\\Desktop\\Parkinson\\lmt'

for i in [ '57', '58']:
    path = basepath + 'patient' + i + '\\'
    rename_face_files(path)
    video_to_FeatureExtraction(path, path_to_openface)
    faceImage(path)
    video_to_FaceLandmarkImg(path, path_to_openface)
    send_lmt_to_LM(path, path_to_RecordPlaybackSample)
    LMJson(path)
