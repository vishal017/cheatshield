import cv2
import logging

def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def preprocess_image(image, size=(24, 24)):
    return cv2.resize(image, size)