import sys

from inference import Inference
from train import Trainer


def run_train_pipeline():
    trainer = Trainer()
    losses = trainer.train()
    print("Model best loss:", losses)


def run_inference_pipeline():
    gray_image_path = './data/inference_image.jpeg'
    inference_pipeline = Inference()
    inference_pipeline.predict(gray_image_path)


if __name__ == "__main__":
    run_type = sys.argv[1]
    print(run_type)
    if run_type == 'train':
        run_train_pipeline()
    else:
        run_inference_pipeline()
