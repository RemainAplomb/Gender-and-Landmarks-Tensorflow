# Gender-and-Landmarks-Tensorflow
## Info
    Notes by: 
        - Rahmani Dibansa
    Github: 
        - https://github.com/RemainAplomb
    Reference Repository for Landmarks:
        - https://github.com/nikhilroxtomar/Human-Face-Landmark-Detection-in-TensorFlow
    Brief Description:
        - This will contain a guide on how to train a Gender and Landmarks model in Tensorflow


## Repository Environment
You can use the notebook to train it in Google Colab. But as time moves forward, Google Colab
updates its python version and tensorflow version. But it should work on Tensorflow 2.x

## Dataset Preparation
To add the annotation for the gender in the Lapa dataset, you should add "-0" or "-1" to
the end of the filename.

Here's an example dataset that I modified from Lapa:
    - https://drive.google.com/file/d/1t2YJft4BOvnwzLl_G790P3uTdjIq0YgO/view?usp=drive_link

## Training Tips
I observed that when both branches are trained at the same time, it gets stuck in a local minima.
To solve this, I only train one branch at a time. To do this, you should freeze a branch in
the build model function.

Here's a sample code snippet:
```
def build_model(input_shape, num_landmarks):
    ...code ommitted...
    # Freeze gender branch
    gender_branch.trainable = False
    for layer in gender_branch.layers:
        layer.trainable = False
    ...code ommitted...
```
and 
```
def build_model(input_shape, num_landmarks):
    ...code ommitted...
    # Freeze landmark branch
    landmark_branch.trainable = False
    for layer in landmark_branch.layers:
        layer.trainable = False
    ...code ommitted...
```

Just experiment on it. Furthermore, you can also try changing the backbone
and whether the backbone is trainable
```
def build_model(input_shape, num_landmarks):
    ...code ommitted...
    backbone = MobileNetV3Large(include_top=False, weights="imagenet", input_tensor=inputs)
    backbone.trainable = True
    ...code ommitted...
```

