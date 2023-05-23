FROM python:3.9

WORKDIR ./

COPY requirements.txt .

RUN pip3 install -r requirements.txt

ADD . C:\Users\Rapat\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\cv2
ADD . C:\Users\Rapat\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\mediapipe


COPY alon.py /

COPY line_boundary_check.py /
COPY main.py /
COPY object_detection.py /
COPY pose_estimation.py /
COPY PoseModule.py /


CMD [ "python", "./alon.py" ]