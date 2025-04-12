# |**Computer Vision**|

_[D.U.Th. semester projects for Computer Vision.]_

# HW1 # Text Recognition using Computer Vision

## Purpose

The goal of this project is to perform text recognition using computer vision. The implementation is done using Python 3.6 and OpenCV v.3.2.17. The following steps are applied to analyze and process an image:

### Workflow

1. **Adding Gaussian Noise**  
   Initially, Gaussian noise is applied to the selected photo that is to be processed and analyzed.

2. **Processing the Original and Noisy Images**  
   For both the original image and the one with noise, the following steps are carried out to achieve the desired results:

---

### 1. Text Line Detection

The program detects all the lines of text in the document. It displays and saves an image where the different lines of text are highlighted. Specifically, for each text line, a bounding box is drawn, and a unique incremental number is assigned to each line.

### 2. Measuring and Displaying Metrics for Each Line

For each detected line, the following metrics are calculated and displayed as program output:

- **a) Text Area**  
  This is the area occupied by the text, defined as the number of pixels that belong to letters (dark pixels) and not the background (light pixels).

- **b) Bounding Box Area**  
  The area of the bounding box that surrounds the line of text.

- **c) Word Count**  
  The number of words contained in the detected line.

- **d) Average Gray Level**  
  The average gray level of the pixels contained within the bounding boxes of the objects, ensuring that the speed of execution is independent of the sub-region size.

---

---

# HW2 # Panorama Generation using OpenCV

## Purpose

The goal of this project is to implement an algorithm that generates panoramas from multiple individual images. The implementation is done using Python 3.6 and OpenCV v.3.2.17. The following tasks are examined:

## Tasks

1. **Panorama Generation from at least Four Images**
   The panorama will be generated by stitching at least four images together using the following feature detectors and descriptors:

   - a) SIFT (Scale-Invariant Feature Transform)
   - b) SURF (Speeded-Up Robust Features)

2. **Evaluation of Panorama Generation Results**
   The results of each generated panorama will be evaluated by calculating the following metrics:

   - **2 - Differential Entropy**
     The differential entropy is calculated as the difference between the average global entropy of the component images and the global entropy of the generated panorama image.

   - **3 - Average Local Entropy for the Stitched Image**
     The local entropy can be computed in sections (neighborhoods) of the image using a sliding window (9x9 pixels). The average local entropy is then calculated for each window position.

   - **4 - Differential Variance of the Local Entropy**
     This is the difference between the local entropy of the panorama image and the local entropy of the component images.

   - **9 - Absolute Difference of Standard Deviations**
     This metric calculates the difference between the standard deviation of the panorama image and the average standard deviation of the component images.

   The entropy-based features (metrics 2-4) can be determined by calculating both the local and global entropy values for the component images and the panorama.

3. **Application of the Algorithm on Four New Images of Any Scene of Interest**
   Implement the panorama generation on four new images from any scene of interest. Perform this process for different camera orientations and varying degrees of scene overlap in consecutive images. Evaluate the generated panoramas using the metrics described in task 2. Conclusions based on the results should be made.

---

# Requirements

- Python 3.6
- OpenCV v.3.2.17

To install the required libraries, you can use pip:

```shell
pip install -r requirements.txt
```

# HW3

# Part A – Multi-class Classification using OpenCV (Bag of Visual Words)

You will implement a Python program using the **OpenCV** library to address the problem of **multi-class image classification**. The implementation includes the following steps:

### 1. Creation of Visual Vocabulary (BoVW using KMeans)

- Generate a visual vocabulary using the **K-Means** clustering algorithm.
- All training images will be used to create this vocabulary.

### 2. Feature Extraction from Training Images

- Extract image descriptors using the **BoVW model**, based on the vocabulary from Step 1.

### 3. Classification of Images Using Two Classifiers

- Using the results of Step 2, classify images using:

#### a) k-Nearest Neighbors (k-NN) algorithm

#### b) One-versus-all SVM:

- One **SVM classifier** will be trained for each class.

### 4. System Evaluation

Using the test dataset, evaluate the **accuracy** of the classification system:

- Measure overall classification accuracy.
- Measure per-class accuracy.
- Evaluate the effect of the following parameters:

#### a) Vocabulary size:

- Use **at least 5 different vocabulary sizes**.
- Select the vocabulary size that gives the **highest accuracy**.

#### b) Number of nearest neighbors (k) in k-NN:

- Find and report the **optimal k value** for this problem.

---

# Part B – CNN Architectures using Keras/Tensorflow

You will implement a **convolutional neural network architecture** (CNN) in Python using **Keras-TensorFlow**, targeting a **multi-class classification** problem.

> All implementations should be in **Jupyter Notebooks**, using **Google Colab**.

You will implement **TWO different architectures**:

### i. Custom CNN (Non Pre-trained)

- A **newly created CNN** architecture designed specifically for this classification task, following lab guidelines.

### ii. Pre-trained CNN

- Choose a **pre-trained network** (e.g., VGG16, ResNet50, MobileNet, etc.).
- Provide a **brief analysis of the selected architecture**.

### For Both Architectures, Include:

**A)** Full description of the architecture and its layers  
**B)** Quantitative evaluation (test set accuracy)  
**C)** Training details & justification:

- Number of epochs
- Input size
- Batch size
- Callbacks (e.g., EarlyStopping)
- Preprocessing steps
- Data augmentation techniques

---

# Part C – Object Detection for Traffic Sign Detection

You will implement a **CNN-based object detection architecture in Python** for detecting **traffic signs**.

> You may use **Keras-TensorFlow** or **PyTorch** libraries.  
> Use **Jupyter Notebook** on **Google Colab** for implementation and training.

You will train and compare **TWO object detection architectures**:

### i. Two-stage Detector: Faster R-CNN

- As presented during lab sessions.

### ii. One-stage Detector: YOLOv3

- You may use a YOLOv3 implementation of your choice (e.g., from `ultralytics`, or your own custom code).

### For Both Approaches, Include:

**A)** Full description of the architecture and layers used  
**B)** Evaluation using **mean Average Precision (mAP)** on the test set  
**C)** Description & justification of training procedure:

- Number of epochs
- Input size
- Batch size
- Callbacks
- Preprocessing
- Data augmentation
