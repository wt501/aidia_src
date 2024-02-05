<div align="center">
  <a href="./README.md">English</a> |
  <a href="./README-ja.md">日本語</a>
</div>

#

<h1 align="center">
  <img src="aidia/icons/icon.png" width="30%"><br>
  Aidia
</h1>

<h4 align="center">
  AI Development and Image Annotation
</h4>

<div align="center">
  <a href="https://www.python.org/">
  <img src="https://img.shields.io/badge/python-3.8_|_3.9-blue?logo=python"></a>
  <a href="https://www.tensorflow.org">
  <img src="https://img.shields.io/badge/TensorFlow-2.10.0-blue?logo=tensorflow"></a>
  <a href="https://www.qt.io">
  <img src="https://img.shields.io/badge/Qt5-5.15.2-blue?logo=qt"></a>
</div>

<br>

<div align="center">
  <img src=".readme/example_01.png" width="100%">
</div>


## Description

Aidia is a medical image annotation tool with AI development utilities.
It is inspired by [Labelme](https://github.com/wkentaro/labelme) and written in Python and uses Qt for its graphical interface.
If you use Aidia without AI utilities on Windows, you can also download Windows binary from [GitHub Releases](https://github.com/wt501/Aidia/releases).


## Features
- Image annotation for polygon, rectangle, polyline, line and point.
- Simply labeling by customized GUI buttons.
- DICOM format support including DICOM files which have no extention (.dcm).
- Adjustment of brightness and contrast by mouse dragging like a DICOM viewer.


## Installation

There are options:

- Platform agnostic installation: [Anaconda](#anaconda)
- Pre-build binaries from [GitHub Releases](https://github.com/wt501/Aidia/releases)

### Anaconda
You need install [Anaconda](https://www.anaconda.com/download), then run below:
```bash
conda create --n aidia python=3.9
conda activate aidia
python install.py
```

## Usage
```bash
conda activate aidia  # if you did not activate the environment.
aidia
```

## GPU Support
TensorFlow GPU support requires below:
- [CUDA Toolkit 11.2](https://developer.nvidia.com/cuda-11.2.2-download-archive)
- [cudDNN 8.1.0](https://developer.nvidia.com/rdp/cudnn-archive)


## How to build standalone executable
Below shows how to build the standalone executable on macOS, Linux and Windows.
```bash
# setup conda
conda create -n aidia_exe python=3.9
conda activate aidia_exe
pip install -r requirement_bin.txt

# build the standalone executable
python build.py
```


## Acknowledgement
This repo is the fork of [wkentaro/labelme](https://github.com/wkentaro/labelme), and uses [ICOOON MONO](https://icooon-mono.com/) for GUI construction.

## References

### YOLO Implementation
[hunglc007/tensorflow-yolov4-tflite](https://github.com/hunglc007/tensorflow-yolov4-tflite)