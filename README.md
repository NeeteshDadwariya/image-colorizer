# Image Colorization Starter Code
The objective is to produce color images given grayscale input image. 

## Setup Instructions
Create a conda environment with pytorch, cuda. 

    `$ conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia`

For systems without a dedicated gpu, you may use a CPU version of pytorch.
`$ conda install pytorch torchvision torchaudio cpuonly -c pytorch`
    
## Dataset
Use the zipfile provided as your dataset. You are expected to split your dataset to create a validation set for initial testing. Your final model can use the entire dataset for training. Note that this model will be evaluated on a test dataset not visible to you.

## Code Guide
Baseline Model: A baseline model is available in `basic_model.py` You may use this model to kickstart this assignment. We use 256 x 256 size images for this problem.
-	Fill in the dataloader, (colorize_data.py)
-	Fill in the loss function and optimizer. (train.py)
-	Complete the training loop, validation loop (train.py)
-	Determine model performance using appropriate metric. Describe your metric and why the metric works for this model? 
- Prepare an inference script that takes as input grayscale image, model path and produces a color image. 

## Additional Tasks 
- The network available in model.py is a very simple network. How would you improve the overall image quality for the above system? (Implement)
- You may also explore different loss functions here.

## Bonus
You are tasked to control the average color/mood of the image that you are colorizing. What are some ideas that come to your mind? (Bonus: Implement)

## Solution
- I have choosen to convert image to L*a*b color space. This is because in L*a*b color space, we can already have the 
  first channel of brightness from the given image, and we only need to predict the rest of two channels.
  But if I had choosen, the model needs to predict 3 channels (R,G and B) which is way difficult as compared to the 2-channel
  approach.

  If I use RGB, I have to first convert image to grayscale, feed the grayscale image to the model and hope it will 
  predict 3 numbers which is a way more difficult and unstable task due to the many more possible combinations of 3 
  numbers compared to two numbers. If we assume we have 256 choices (in a 8-bit unsigned integer image this is the 
  real number of choices) for each number, predicting the three numbers for each of the pixels is choosing between 
  256³ combinations which is more than 16 million choices, but when predicting two numbers we have about 65000 choices

- Model Performance:
  - Model performance can be measured in the terms of MSE loss between (target_ab, predicted_ab) channels of the image.
  - This is suitable error metric, as it will try to bring the model weights closer to the original image pixel values.

- Inference Script:
  - Inference script has been implemented in the file `inference.py` and can be run via main command. (Details below.)

- Tried things that worked, and which did not - 
  - Able to implement the whole train and inference pipeline with the required details.
  - Since new to PyTorch, faced difficulties while setting up the whole pipeline.
  - Using ResNet architecture and sigmoid output.
  - Model losses are approx ~680. Output images are partially colored. Could not test on full dataset.

- How to run the program - 
  - Install all dependencies present in environment.yml using `conda env create -f environment.yml`
  - Please note the data should be present in the following tree:
    - All train images at: `./data/input/train/`
    - All validation images at: `./data/input/val/`
    - The output images will go at: `./data/output/`
    - Best model is saved at - `./checkpoints/best_model.tar`
  - To run the train pipeline, use command - `python main.py train`.
  - To run the inference pipeline, use command - `python main.py inference`
