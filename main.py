"""
Created on Tuesday, January 22 of 2024 by Greg
"""
import os
# import PySpin
import matplotlib.pyplot as plt
import keyboard
import time
import numpy as np
import mysql.connector
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

global continue_recording
continue_recording = True
class camera:
    def __init__(self):

        mydb = mysql.connector.connect(
            host="localhost",
            user="Greg",
            password="contpass01",
            database="AIRY"
        )

        mycursor = mydb.cursor()

newCam = camera()
print(continue_recording)