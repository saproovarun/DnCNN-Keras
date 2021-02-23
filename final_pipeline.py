import sys
import tensorflow as tf
import numpy as np
import cv2
import time
import tensorflow_model_optimization as tfmot
import tflite_runtime.interpreter as tflite
import matplotlib.pyplot as plt
import os
import glob
from skimage.metrics import peak_signal_noise_ratio
import torch
import torch.nn as nn
import seaborn as snb
import re
from torchvision.io import read_video
from skvideo.io import FFmpegWriter


class Denoiser:
    def __init__(self, merge_outputs):
        self.model = tf.keras.models.load_model('./model')
        self.merge_outputs = merge_outputs

    def get_patches(self, frame):
        patches = np.zeros(shape=(self.batch_size, 50, 50, 3))
        counter = 0
        for i in range(0, self.SCALE_H, 50):
            for j in range(0, self.SCALE_W, 50):
                patches[counter] = frame[i:i+50, j:j+50]
                counter+=1
        return patches.astype(np.float32)
        
    def reconstruct_from_patches(self, patches, h, w, true_h, true_w, patch_size):
        img = np.zeros((h,w, patches[0].shape[-1]))
        counter = 0
        for i in range(0,h-patch_size+1,patch_size):
            for j in range(0,w-patch_size+1,patch_size):
                img[i:i+patch_size, j:j+patch_size, :] = patches[counter]
                counter+=1
        return cv2.resize(img, (true_w, true_h), cv2.INTER_CUBIC)
    
    def denoise_video(self, PATH):
        self.cap = cv2.VideoCapture(PATH)
        self.H, self.W = int(self.cap.get(4)), int(self.cap.get(3))
        self.SCALE_H, self.SCALE_W = (self.H//50 * 50), (self.W//50 * 50)
        self.batch_size = ((self.SCALE_H * self.SCALE_W) // (50**2))
        
        outputFile = './denoise.mp4'
        writer = FFmpegWriter(
        outputFile,
            outputdict={
            '-vcodec':'libx264',
            '-crf':'0',
            '-preset':'veryslow'
        }
        )
        
        while True:
            success, img = self.cap.read()
            if not success:
                break
            resize_img = cv2.resize(img, (self.SCALE_W, self.SCALE_H), cv2.INTER_CUBIC).astype(np.float32)

            noise_img = resize_img/255.0
            patches = self.get_patches(noise_img).astype(np.float32)
            predictions = np.clip(self.model(patches), 0, 1)
            pred_img = (self.reconstruct_from_patches(predictions, self.SCALE_H, 
                                                     self.SCALE_W, self.H, self.W, 50)*255.0)
            if self.merge_outputs:
                merge = np.vstack([img[:self.H//2,:,:], pred_img[:self.H//2,:,:]])
                writer.writeFrame(merge[:,:,::-1])
            else:
                writer.writeFrame(pred_img[:,:,::-1])
        writer.close()
        
PATH = sys.argv[1]
print(f"Path is : {PATH}")
denoise = Denoiser(merge_outputs = True)
x = denoise.denoise_video(PATH)

