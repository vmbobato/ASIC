# ASIC - AI for Satellite Image Classification WebApp

This project is a web-based application that allows users to upload satellite images and receive a **segmented output** using [Meta's SAM2](https://github.com/facebookresearch/segment-anything) model. It simulates an Earth observation tool where land cover types can be detected and visualized.

> Future versions will classify each segmented region using a lightweight image classification model (e.g., ResNet50, MobileNetV2).

---

## Features

- Upload a satellite image  
- Segment regions using **SAM2**  
- Visualize the output overlaid on the image  
- *(WIP)* Classify each region using a trained model  

---

## Technologies Used

### Backend
- **Flask** – lightweight Python web server  
- **PyTorch** – to load and run the SAM2 model  
- **SAM2** – automatic mask generator for segmentation  
- **OpenCV**, **NumPy**, **Pillow** – image handling  

### Frontend
- **HTML/CSS/JS** – styled with a satellite-dashboard theme  
- **Dynamic UI** – preview both uploaded and segmented images  

### ML Models
- **SAM2** – from `segment-anything` by Meta  
- **(Optional)** TensorFlow/Keras for region classification  


