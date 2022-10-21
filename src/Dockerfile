FROM nvcr.io/nvidia/pytorch:21.02-py3
COPY detect-waste /detect-waste
ENV DEBIAN_FRONTEND=noninteractive
RUN cd /detect-waste/efficientdet && pip install -r requirements/requirements.txt
RUN pip uninstall -y opencv-python
RUN pip uninstall -y opencv-contrib-python
RUN pip uninstall -y opencv-contrib-python-headless
RUN apt-get update -y
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install opencv-python==4.5.5.64

