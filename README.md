# Goal
Design and implement devanagari character recognition using Convolutional Neural Networks.

# Dataset 
Dataset is obtained from https://www.kaggle.com/rishianand/devanagari-character-set

The dataset consists of 92000 rows (92 thousand sample images), and 1025 columns. Each row contains the pixel data ("pixel0000" to "pixel1023"), in greyscale values (0 to 255). The column "character" represents the Devanagari character name corresponding to each image.

## To run the code on a computer (with CPU)

### Git clone the code
```
git clone https://github.com/ArbinTimilsina/DevanagariCharacterRecognization.git
cd DevanagariCharacterRecognization

Put the data.cvs file from the dataset inside input_files.
```

### Create a conda environment (Python 3)
```
conda create --name envDevRecognization python=3.5
conda activate envDevRecognization
pip install --upgrade pip
pip install -r requirements/cpu_requirements.txt
```

### Switch Keras backend to TensorFlow
```
KERAS_BACKEND=tensorflow python -c "from keras import backend"
```

### Create an IPython kernel for the environment
```
python -m ipykernel install --user --name envDevRecognization --display-name "envDevRecognization"
```

###  To open jupyter notebook, do
```
jupyter notebook model_training.ipynb
```
Make sure to change the kernel to envDevRecognization using the drop-down menu (Kernel > Change kernel > envDevRecognization)
