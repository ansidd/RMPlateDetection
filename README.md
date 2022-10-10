# Robomaster Plate Detection

This project is a armor plate locator module in the context of the [Robomaster](https://www.robomaster.com/en-US/robo/rm?djifrom=nav) competition.

An object detection model is used to detect the armor plates on the robots inorder to attack the enemy robots.

The following important exercises were performed :

1. Preparing data: The data available at the start of the process was [this](https://drive.google.com/drive/folders/1w4KQpC82ST2jVguzBdu4wQb6ceWD1dZf) which are a bunch of images from gameplay with annotations that included the locations of the armor plates in the images stores in .xml files. Since the algorithm we are using to perform object detection is YOLO, the data was converted to a format that is easier to load and train the model. This is done with the code in "PlateDetection.ipynb" notebook.

2. Training YOLO: Object detection is done using the yolov5 model which is a state-of-the-art object detection model with a very smal inference runtime. I followed this [tutorial](https://kikaben.com/yolov5-transfer-learning-dogs-cats/) to train the pretrained yolov5 model to learn to detect armor plates.

3. Inference on video: After training the model for a certain number of epochs, the model with the best weights is stores as "best.pt". These weights are loaded and are used to perform detection. The detection code is in "main.py". Here a video "vid.mp4" which is a video of the match between NYU and CU Boulder is read and the prediction is done on each frame of the video and displayed simultaneously.

Some stills from the detection:
![image]{width: 100px;}(https://github.com/ansidd/RMPlateDetection/raw/main/Screen%20Shot%202022-10-10%20at%204.11.42%20PM.png)

