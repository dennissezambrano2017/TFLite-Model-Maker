# FTC-TFLite-Model-Maker
This repo is for creating a TFLite Object Detection Model to be used in the [FIRST Tech Challenge (FTC) competition for 2024](https://firstroboticsbc.org/ftc/centerstage-season/).

## Google Colab
The jupyter notebook in this repo is made to be used in [Google Colab](https://colab.research.google.com/).

**[Please use this version hosted on Google Colab.](https://colab.research.google.com/drive/1U4g6Lf_PZ9DSS43DDHdqTWQ3WGQyTAUn?usp=sharing)**

## Model Tester
Included in this repo is a model tester. This tester needs to be run on your own computer and cannot be run in Google Colab as it requries access to your webcam's live video. Colab is not capable of this.

The model tester can be used to quickly test your model without needing to upload it to your robot controller, without programming any autonomous code, or testing with your robots webcam. Just see how well your model can recognize your trained objects.

To run the model tester:
1. Put your model in the same folder with the `tester.py` file.
1. Open your shell environment / terminal in the model tester directory.
1. Install depedancies using the following command in your shell environment:

   `pip install -r requirements.txt`
   
1. Run the tester. 
    * You **must** give it the filename of your model as the first arument
    * You can *optionally* give it a detection threshold between 0.00 and 1.00 (0% - 100%). This argument determins the minimum inference score required for the program to display bounding boxes of what it has detected using your model. If you don't provide one, it will default to 0.50.

    `python tester.py model.tflite 0.50`