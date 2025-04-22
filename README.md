# ASIC - AI for Satellite Image Classification WebApp

This project is a web-based application that allows users to upload satellite images and receive a **segmented output** using [Segformer B2 ADE 20K](https://huggingface.co/nvidia/segformer-b2-finetuned-ade-512-512) model. It simulates an Earth observation tool where land cover types can be detected and visualized.

> This work was also supported by Andreas Bardram and Theo Lin.

---

## Features

- Upload a satellite image  
- Segment regions using **SegFormerB2**  
- Visualize the output overlaid on the image  
- Classify each region using a pretrained model
- Calculate each segmented area and represent them in a percentage.

---

## Technologies Used

### Backend
- **Flask** – lightweight Python web server  
- **PyTorch** – to load and run the SAM2 model  
- **Segformer B2** – automatic mask generator for segmentation  
- **OpenCV**, **NumPy**, **Pillow** – image handling  

### Frontend
- **HTML/CSS/JS** – styled with a satellite-dashboard theme  
- **Dynamic UI** – preview both uploaded and segmented images  

### ML Models
- **Segformer B2** – from by Nvidia  
- **PyTorch & Torchvision** - for model implementation

## Steps to run ASIC
Clone this repository into your desired location.

`git clone https://github.com/ASIC.git`
`cd ASIC/`
`pip install -r requiremnts.txt`
`python3 app.py`






