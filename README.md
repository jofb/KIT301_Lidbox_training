# KIT301 Model Training
This project contains the necessary files to train a spoken language identification model. It requires some setup before it will work correctly.

This project is first and foremost adapted from https://github.com/py-lidbox/lidbox.

## Environment Setup
1. Clone repo with `git clone github`
2. Setup virtual environment 
3. Install requirements with `pip install -r requirements.txt`
4. Install TensorFlow separately with `pip install tensorflow`

## Project Setup
Make sure to change the input and output directories to match your setup. The input directory structure should have a folder for each dataset which contains the metadata for the dataset, and a folder called clips with each audio clip.

Additionally, change the `languages` string to match the language codes that you are using, these are the labels the model will use for training on the dataset.

At this point the main file can be correctly run to train a model. The output model and checkpoints will be nested away in the output directory.

## Notes
`dataset_processing` contains methods for manipulating the datasets to be fit for batch processing. I don't fully understand every method in here, it is a lot of pandas manipulation.

`audio_processing` contains all the methods for the audio processing pipeline. 
To correctly process the audio, first the audio signal must be read in as a float32 array. Then, its resampled to 16kHz (so that it isn't too big). 
The signal is then filtered, removing silence using voice activity detection.
A spectrogram is created on the signal, and then this is converted to Mel scale, and finally, log scale.
The result is a log Mel spectrogram for each audio file.

The neural network is then trained on the spectrograms.
 
