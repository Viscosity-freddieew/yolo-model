# real-time-night-vision-pothole-detection-on-raspberry-pi
Pothole detection from real time video stream or images with Python on a Raspberry Pi 4 Model B 8gb Ram

- Change the python version to 3.9.18
```
sudo apt update
sudo apt install -y make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
    libffi-dev liblzma-dev

cd /usr/src
sudo wget https://www.python.org/ftp/python/3.9.18/Python-3.9.18.tgz
sudo tar xzf Python-3.9.18.tgz
cd Python-3.9.18

sudo ./configure --enable-optimizations
sudo make -j$(nproc)
sudo make altinstall

python3.9 --version

sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
sudo update-alternatives --config python3

sudo update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.9 1
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 2

sudo update-alternatives --config python3

python3 --version
```
- Create Virtual Environment
```
virtualenv -p python3 yolo-env
source yolo-env/bin/activate

```
- Enter cloned folder
```
cd yolo-model
```
- Download Ultralytics
```
pip install ultralytics
```
- Download Torch, Torchvision, Opencv, Numpy, FFmpeg
```
bash -c "pip install Cython==0.29.36"

# install the packages using standard pip3 install
packages=('scipy==1.5.2' 'pandas==1.1.3' 'librosa==0.8.0' 'tqdm==4.54.1' 'pytorch-lightning==1.1.6' 'tensorboardx==2.1' 'pyyaml==5.3.1' 'munch==2.5.0' 'fire==0.3.1' 'ipython==7.19.0')

for i in "${packages[@]}"; do
    pip install $i
done

# Install the dependencies
pip3 install setuptools==58.3.0

#Torch & Torchvision
pip install torch -f https://torch.kmtea.eu/whl/stable.html
pip install torchvision -f https://torch.kmtea.eu/whl/stable.html

#opencv
pip install opencv-python opencv-contrib-python

#FFmpeg
sudo apt update
sudo apt install -y libopencv-dev libglib2.0-dev libgtk-3-0 ffmpeg

#Numpy
pip3 install --upgrade --force-reinstall numpy

sudo apt-get remove --purge python3-numpy
pip3 uninstall numpy
sudo apt-get install python3-numpy

sudo apt-get install python3-numpy

```
- Run the code with command mentioned below. (You can change my_model6 to any other yolo model, it should still work the same. P.S. The my_model7.pt is trained on more pothole images.)

```
python3 yolo_detect3.py --model my_model6.pt --resolution 640x480

```
# To train your own YOLO Model
- Download Label Studio on your computer/pc
```
On your terminal:

Conda Create --name yolo-env1 python=3.12
conda activate yolo-env1

pip install label-studio

label-studio start

```
- On label studio, go to signups (Create a email address and password), create your project; upload your dataset and annotate. Once your are done export the dataset as YOLO
- Follow this tutorial: https://www.youtube.com/watch?v=r0RspiLG260&t=442s
- Next transfer your dataset to this https://colab.research.google.com/github/EdjeElectronics/Train-and-Deploy-YOLO-Models/blob/main/Train_YOLO_Models.ipynb#scrollTo=qcBdnOA9v85S, follow the instructions to create your own YOLO model (You can change the yolo model type based on your needs)

# Using your own model
- Transfer your own YOLO model into the raspberry pi (either through github or ssh)
- Transfer the your_model.pt into yolo-model folder of the raspberry pi or copy the yolo_detect3.py codes onto the folder where your_model.pt is
- Run this code
```
python3 yolo_detect3.py --model your_model.pt --resolution 640x480

```
  
# Change the code according to the type of camera that you used
- For this project we used gstreamer and opencv to watch the video feed as it uses a Pi Night-Vision Camera (For me there was issues with installing Picamera2 as I'm using Debian Bookwarm)
- If you run into any issues, screenshot or copy the error Chatgpt to clarify and solve.
- If the code doesnt work, it could be opencv issue, gstreamer installation issue, ffmeg issue.
- my_model is am YOLOv11 model and my_model5 is a yolov5 model (both work quite poorly here)
- my_model6 is a Yolov8n model trained under 320x320 resolution, while my_model2 is a Yolov8s model trained under 640x640 resolution (This a better at detection but very low fps compared to my_model6)

# Example images
![image](https://github.com/user-attachments/assets/7f306bfb-03af-4713-be3a-6639a54239ee) ![image](https://github.com/user-attachments/assets/fb73c797-7367-4031-91d0-caf3cb12d88c)
![image](https://github.com/user-attachments/assets/171605cb-f68a-4c3d-b7f0-e9abf73e6307)


