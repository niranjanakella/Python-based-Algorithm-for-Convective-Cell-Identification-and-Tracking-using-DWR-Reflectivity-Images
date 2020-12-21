# Python-based-Algorithm-for-Convective-Cell-Identification-and-Tracking-using-DWR-Reflectivity-Images
The identification and tracking of these Convective Cells through out their lifetime are of great significance in the weather and climate system. The present study developed an autonomous Python-based algorithm for the identification and tracking of Convective Cells using Doppler Weather Radar (DWR) reflectivity images and the preliminary results are discussed. The proposed algorithm implements Deep Neural Network-based Computer Vision (CV) approach for the identification and tracking of Convective Cells using Optical Character Recognition (OCR) engine "Tesseract"

# Package Installation & Algorithm Execution
### Anaconda Environment: (Recomended)
<hr>

- Install Anaconda software : <a href ='https://www.anaconda.com/products/individual#windows'>From Here</a>
- After Installation open Anaconda Prompt (Anaconda Command Line)
- Type the following commands in the command line to install all the necessary libraries:
```
conda install -c conda-forge opencv
conda install -c conda-forge imutils
conda install -c anaconda numpy
conda install -c conda-forge pytesseract
conda install -c conda-forge scikit-image
conda install -c anaconda scipy
conda install -c conda-forge argparse
```
- Download the tesseract OCR through <a href='http://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-setup-4.00.00dev.exe'> this </a>link. Path to executable file: 
  - Windows:  "C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe"
  - MacOS & Linux: "/usr/local/bin/tesseract" or Find the .exe by typing ```sudo find / -name "tessdata"``` on terminal.

- Use the command line to execute the python script with the necessary arguments:
  - To run Algorithm 1 - "TwoImages_ConvectiveCellTracking" execute the following command.
```
run TwoImage_AnomalyTracking.py –image1 PathToImage1 –image2 PathToImage2
```
- - To run Algorithm 2 - "FolderOfImages_ConvectiveCellTracking" execute the following command.
```
run FolderOfImages_AnomalyTracking.py -f PathToFolderContainingAllImages
```

### Other Recommended Methods
<hr>

1. Google Colab
- Kindly use the following code to install necessary packages and to mount your drive.
```
!pip install pytesseract
!sudo apt install tesseract-ocr
from google.colab import drive
drive.mount('/content/drive')
```
2. Other IDEs for Python


