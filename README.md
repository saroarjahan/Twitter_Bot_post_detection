# Twitter Bot Post Detection

This repository contains a trained model for detecting Twitter bot posts, akin to the functionality of Botometer. Given that Botometer is a premium service requiring payment for API access, this project has been developed as an open-source alternative.

Our model has been trained on 50,000 bot and genuine posts using state-of-the-art machine learning technologies, such as OpenAI and BERT. It has achieved an overall accuracy of 87% in bot detection.

## Dataset

The dataset used for training was sourced from open-access resources, and benchmarked data was employed for testing purposes. One such dataset used was TwiBot-20.

Reference papaer: 

@inproceedings{feng2021twibot,
  title={Twibot-20: A comprehensive twitter bot detection benchmark},
  author={Feng, Shangbin and Wan, Herun and Wang, Ningnan and Li, Jundong and Luo, Minnan},
  booktitle={Proceedings of the 30th ACM International Conference on Information \& Knowledge Management},
  pages={4485--4494},
  year={2021}
}

## Model Access

The trained model, `model_1.pt`, is available for download from Google Drive via the following [link](https://drive.google.com/file/d/1pxeexKiZmSFRNvg940fSc4AhptaM9mka/view?usp=sharing). Should you encounter any issues in accessing the model, please feel free to contact me via email at saroarjahan.bd@gmail.com.

## Usage Instructions

To use the model, run the `model.py` or `model.ipynb` file, and make sure u have downloawd model_1.pt bfeore running it.

## Required Libraries

You may need to install the following libraries:

1. `torch`: Install using `pip install torch`
2. `transformers`: Install using `pip install transformers`
3. `pandas`: Install using `pip install pandas`

These libraries are necessary for running the model and processing data.
