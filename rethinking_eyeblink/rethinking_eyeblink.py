# Rethinking Eye-blink project

## Author(s): Dr. Youngjun Cho *(Assistant Professor, UCL Computer Science)
## * http://youngjuncho.com

# This example monitors only a user's left eye. Please contact the author if you have a question of how to get other areas.
# Note that this code uses the Dlip library and a pre-trained model available at dlib.net (for face landmark detection)
# Please check: http://dlib.net/face_detector.py.html

# Note that:
# Some parts are not optimized given our "demo" purpose - enjoy Rethinking Eye-blink!
# There is a need for improving the ways to use a pretrained model for real-time task difficulty estimation
# and to make it adapt to each situation and context (e.g. retraining a pretrained model)

import cv2
import numpy as np
import os, time
from imutils import face_utils # These are used for a quick demo - not optimized.
import dlib
from rethinking_eyeblink.utils.cal_ratio_size import cal_ratio_size
from rethinking_eyeblink.utils.blink_spectrogram import blink_spectrogram
import pandas as pd
from torchvision import transforms # These are used for a quick demo - not optimized.
from torchvision.transforms import Resize, ToTensor # These are used for a quick demo - not optimized.
from PIL import Image # These are used for a quick demo - not optimized.
from tensorflow import keras
import threading

# import scipy.io as sio
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # Just in case you have an issue with Tensorflow - AVX2 FMA

class rethinking_eyeblink:

    def __init__(self, cam_resolution=480, spectrogram_width=260, pretrained_model_path_name='model.h5', update_rate=5 ):
        self.spectrogram_width=spectrogram_width # unit: second
        self.update_rate = update_rate # unit = second
        self.CameraResolution=cam_resolution
        self.fps = 20 # Only for the real-time demo purpose. Higher fps is preferable for post-processing
        self.blw_removal_size = self.fps
        # self.pretrained_model = keras.models.load_model(pretrained_model_path_name)
        self.pretrained_model_path = pretrained_model_path_name



    def time_in_milli(self):
        return round(time.time() * 1000)
    # Example code for checking time delay
    #                 c_time = self.time_in_milli()
    #                 print("1-time:"+str(c_time))



    def rethinking_eyeblink_run(self, MODE=1, VIEWMODE=2, filepath_name='./your_video_file.mp4'):

        # We use the Dlip library and a pre-trained model available at dlib.net (for face landmark detection)
        # http://dlib.net/face_detector.py.html
        face_detector = dlib.get_frontal_face_detector()
        currentpath = os.getcwd()
        landmark_prediction_model = dlib.shape_predictor(currentpath+'/rethinking_eyeblink/model/shape_predictor_68_face_landmarks.dat')
        cv2.namedWindow('Rethinking Eye-blink', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Rethinking Eye-blink2', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Rethinking Eye-blink spectrogram', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Rethinking Eye-blink estimation-bar', cv2.WINDOW_NORMAL)


        # Initial frame preparation

        if MODE ==1:
            vidcap = cv2.VideoCapture(0)
        else:
            vidcap = cv2.VideoCapture(filepath_name)

        success, imgframe = vidcap.read()

        if self.CameraResolution == 360:
            imgframe = cv2.resize(imgframe, (640, 360))
        elif self.CameraResolution == 480:
            imgframe = cv2.resize(imgframe, (720, 480))


        eye_ratio_seq = []
        eye_area_size_seq = []
        LEFT_EYE_POINTS = [36, 37, 38, 39, 40, 41]

        insidecount = 0
        imshow_update_count=0
        prev_time = 0

        while success:

            time_elapsed = time.time() - prev_time
            if time_elapsed > 1./self.fps:

                # the processing below takes relatively long (on Macbook Pro 2018 - 30ms)
                imgframe = imgframe.astype('uint8')
                faces = face_detector(imgframe)


                if np.shape(faces)[0] == 0:
                    print("There is no face detected")
                    eye_ratio=np.nan
                    eye_area_size=np.nan

                imEye_fixed_size=[]
                eye_detected =False


                for face in faces:
                    model_output = landmark_prediction_model(imgframe, face)

                    # Masking - removing non-eye areas
                    model_output_np = face_utils.shape_to_np(model_output)
                    lefteye = model_output_np[LEFT_EYE_POINTS]
                    min_yx = np.min(lefteye, axis=0)
                    max_yx = np.max(lefteye, axis=0)

                    remapped_eye = cv2.convexHull(lefteye)

                    face_zeros = np.zeros_like(imgframe)
                    for (x, y) in model_output_np:
                        cv2.circle(face_zeros, (x, y), 1, (0, 0, 255), -1)
                    eye_mask = np.zeros((imgframe.shape[0], imgframe.shape[1]))

                    cv2.fillConvexPoly(eye_mask, remapped_eye, 1)
                    eye_mask = eye_mask.astype(np.bool)
                    face_zeros[eye_mask] = imgframe[eye_mask]

                    only_eye = face_zeros[
                               int(min_yx[1] / 2 + max_yx[1] / 2) - int((max_yx[0] - min_yx[0]) / 2):int(
                                   min_yx[1] / 2 + max_yx[1] / 2) + int((max_yx[0] - min_yx[0]) / 2),
                               min_yx[0]:max_yx[0],
                               :]
                    imEye_fixed_size = cv2.resize(only_eye, (200, 200))
                    eye_detected=True


                    _t = (int((model_output.part(LEFT_EYE_POINTS[1]).x + model_output.part(LEFT_EYE_POINTS[2]).x) / 2),
                          int((model_output.part(LEFT_EYE_POINTS[1]).y + model_output.part(LEFT_EYE_POINTS[2]).y) / 2))
                    _b = (int((model_output.part(LEFT_EYE_POINTS[5]).x + model_output.part(LEFT_EYE_POINTS[4]).x) / 2),
                          int((model_output.part(LEFT_EYE_POINTS[5]).y + model_output.part(LEFT_EYE_POINTS[4]).y) / 2))
                    _l = (model_output.part(LEFT_EYE_POINTS[0]).x, model_output.part(LEFT_EYE_POINTS[0]).y)
                    _r = (model_output.part(LEFT_EYE_POINTS[3]).x, model_output.part(LEFT_EYE_POINTS[3]).y)

                    eye_ratio, eye_area_size = cal_ratio_size(_t, _b, _l, _r)

                    break #limiting only one face.


                if not eye_detected:# When eyes are not detected, force to RESET for better calculation
                    eye_ratio_seq=[]
                    eye_area_size_seq =[]
                    print("Eyes are NOT detected - We are forcing to reset eye-blink collection for better calculation")
                else:
                    eye_ratio_seq.append(eye_ratio)
                    eye_area_size_seq.append(eye_area_size)


                if len(eye_ratio_seq)>=int(self.spectrogram_width *self.fps)+ self.blw_removal_size:

                    insidecount += 1
                    if insidecount%(self.fps*self.update_rate)==0:
                        thread_y=threading.Thread(target=self.auto_assessment_caller, args=(eye_ratio_seq,))
                        thread_y.start()
                        insidecount =0

                    eye_ratio_seq.pop(0)



                # Visualization:
                if VIEWMODE == 1 and eye_detected:  # Realtime - This makes it very slow - so just use this when your machine is very powerful or for the offline purpose
                    # c_time = self.time_in_milli()
                    # print("1-time:" + str(c_time))
                    thread_y = threading.Thread(target=self.sequence_show, args=('Rethinking Eye-blink', imEye_fixed_size,))
                    thread_y.start()
                    thread_y = threading.Thread(target=self.sequence_show,
                                                args=('Rethinking Eye-blink2', face_zeros,))
                    thread_y.start()
                    if cv2.waitKey(1) == 27:
                        break
                    # c_time = self.time_in_milli()
                    # print("2-time:" + str(c_time))
                elif VIEWMODE == 2 and eye_detected:  # Demo purpose
                    imshow_update_count+=1
                    your_update_rate_coef=3
                    if imshow_update_count>your_update_rate_coef:
                        thread_y = threading.Thread(target=self.sequence_show,
                                                    args=('Rethinking Eye-blink', imEye_fixed_size,))
                        thread_y.start()
                        thread_y = threading.Thread(target=self.sequence_show,
                                                    args=('Rethinking Eye-blink2', face_zeros,))
                        thread_y.start()

                        imshow_update_count=0
                        if cv2.waitKey(1) == 27:
                            break


                # Next frame
                success, imgframe = vidcap.read()
                if self.CameraResolution == 360:
                    imgframe = cv2.resize(imgframe, (640, 360))
                elif self.CameraResolution == 480:
                    imgframe = cv2.resize(imgframe, (720, 480))


        vidcap.release()
        cv2.destroyAllWindows()

        return (eye_ratio_seq, eye_area_size_seq)

    def sequence_show(self, window_name, data):
        cv2.imshow(window_name, data)

    def auto_assessment_caller(self, eye_ratio_seq):
        # c_time = self.time_in_milli()
        # print("1-time:" + str(c_time))

        df = pd.DataFrame(eye_ratio_seq, columns=['blinkraw'])
        df.insert(len(df.columns), "smooth", df['blinkraw'].ewm(span=self.blw_removal_size, adjust=False).mean())
        df.insert(len(df.columns), "blw_removal", df['blinkraw'] - df['smooth'])

        b_spec = blink_spectrogram(self.fps, df['blw_removal'].values[self.blw_removal_size:])

        composed = transforms.Compose([Resize(size=(93, 200)),
                                       ToTensor()])
        o_spec = composed(Image.fromarray(b_spec))
        o_spec = np.transpose(o_spec.data.numpy(), (1, 2, 0))
        o_spec = o_spec[:, :, 0]
        norm_spec = np.zeros((93, 200))
        final_spec = cv2.normalize(o_spec, norm_spec, 0, 1, cv2.NORM_MINMAX)
        cv2.imshow('Rethinking Eye-blink spectrogram', final_spec)
        # For MATLAB users, you can load your spectrogram in Matlab
        # sio.savemat('rethinking_eyeblink.mat', {'spec': final_spec, 'raw':df['blw_removal'].values[self.blw_removal_size:]})

        spec_reshape = np.reshape(final_spec,
                                  (1, np.shape(final_spec)[0], np.shape(final_spec)[1]))
        spec_reshape = np.transpose(spec_reshape, (0, 2, 1))

        pretrained_model = keras.models.load_model(self.pretrained_model_path)
        pred_output = pretrained_model.predict(spec_reshape)[0]

        estimation_display_bar = np.zeros((93, 240, 3))
        estimation_display_bar[:, 0:int(240 * pred_output[2]), 2] = 1
        estimation_display_bar = cv2.putText(estimation_display_bar,
                                             'Estimated: ' + str(int(pred_output[2] * 100)) + "%", (10, 54),
                                             cv2.FONT_HERSHEY_SIMPLEX,
                                             0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Rethinking Eye-blink estimation-bar', estimation_display_bar)

        # c_time = self.time_in_milli()
        # print("2-time:" + str(c_time))