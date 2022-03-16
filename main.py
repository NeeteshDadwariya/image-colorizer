from inference import Inference
from train import Trainer


def run_train_pipeline():
    trainer = Trainer()
    trainer.train()


def run_inference_pipeline():
    gray_image_path = './data/inference_image.jpeg'
    inference_pipeline = Inference()
    inference_pipeline.predict(gray_image_path)


if __name__ == "__main__":
    run_train_pipeline()
    run_inference_pipeline()
