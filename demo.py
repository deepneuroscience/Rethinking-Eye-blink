# Rethinking Eye-blink project

## Author(s): Dr. Youngjun Cho *(Assistant Professor, UCL Computer Science)
## * http://youngjuncho.com

# [Example DEMO]

# Note that:
# Some parts are not optimized given our "demo" purpose - enjoy Rethinking Eye-blink!
# There is a need for improving ways to use a pretrained model for real-time task difficulty estimation
# and to make it adapt to each situation and context (e.g. retraining a pre-trained model)


from rethinking_eyeblink.rethinking_eyeblink import rethinking_eyeblink

'''
Input args for initialization:
+ CameraResolution=480(p) or 360(p)
+ spectrogram_width=260 (seconds)
+ pretrained_model_name='your_model.h5'
+ update_rate=10 (seconds) - update rate for spectrogram and difficulty estimation
'''

# Although here we offer some examples of pretrained models for testing (in the 'model' directory),
# we encourage you to train your own model and use it,
model_run = rethinking_eyeblink(360, 260, './rethinking_eyeblink/model/rethinking_eyeblink_model_b.h5', 10)

'''
RUN Rethinking Eye-blink framework:
+ MODE = 1 (using a webcam for a realtime demo) or 2 (using a recorded video file)
  For MODE 2, you should add the last argument, eg. filepath_name='./your_video_file.mp4'
+ VIEWMODE = 1 (presenting every single frame - requiring heavy resources) or 2 (picking some frames to demo)
'''
model_run.rethinking_eyeblink_run(MODE=1,VIEWMODE=2)

