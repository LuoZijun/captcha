#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import glob
import pickle
import signal
import warnings
import logging

logger = logging.getLogger("captcha")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter("ignore")

import cv2
import imutils
import imutils.paths
import numpy as np
import tensorflow as tf

from PIL import Image, ImageFilter

from keras.models import Sequential
from keras.models import load_model
from keras.layers.core import Flatten, Dense
from keras.layers.convolutional import Conv2D, MaxPooling2D
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

"""
验证码相关文档以及博客:

How to break a CAPTCHA system in 15 minutes with Machine Learning:
    https://medium.com/@ageitgey/how-to-break-a-captcha-system-in-15-minutes-with-machine-learning-dbebb035a710
python验证码识别实战:
    https://www.hi-roy.com/2017/12/29/python%E9%AA%8C%E8%AF%81%E7%A0%81%E8%AF%86%E5%88%AB%E5%AE%9E%E6%88%98/
python验证码识别6:kNN:
    https://www.hi-roy.com/2017/10/14/Python%E9%AA%8C%E8%AF%81%E7%A0%81%E8%AF%86%E5%88%AB6/
python验证码识别1:灰度处理、二值化、降噪、tesserocr识别:
    https://www.hi-roy.com/2017/09/19/Python%E9%AA%8C%E8%AF%81%E7%A0%81%E8%AF%86%E5%88%AB/
python验证码识别3:滑动验证码:
    https://www.hi-roy.com/2017/09/21/Python%E9%AA%8C%E8%AF%81%E7%A0%81%E8%AF%86%E5%88%AB3/
python验证码识别4:滴水算法分割图片:
    https://www.hi-roy.com/2017/09/22/Python%E9%AA%8C%E8%AF%81%E7%A0%81%E8%AF%86%E5%88%AB4/
fuck_sjtu_captcha:
    https://github.com/shenyushun/fuck-captcha/blob/master/fuck_sjtu_captcha.py
"""

SCRIPT_DIR   = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = os.path.realpath(os.path.join(SCRIPT_DIR, "./"))

CAPTCHA_CHARS_LEN     = 4
SAMPLES_DIR           = os.path.join(PROJECT_ROOT, "data/samples")
EXTRACTED_DIR         = os.path.join(PROJECT_ROOT, "data/extracted")
MODEL_FILENAME        = os.path.join(PROJECT_ROOT, "data/captcha_model.hdf5")
MODEL_LABELS_FILENAME = os.path.join(PROJECT_ROOT, "data/model_labels.dat")


def resize_to_fit(image, width, height):
    (h, w) = image.shape[:2]
    if w > h:
        image = imutils.resize(image, width=width)
    else:
        image = imutils.resize(image, height=height)
    padW = int((width - image.shape[1]) / 2.0)
    padH = int((height - image.shape[0]) / 2.0)
    image = cv2.copyMakeBorder(image, padH, padH, padW, padW, cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))

    return image

def binarizing(img, threshold):
    # 灰度处理
    img = img.convert("L")
    pixdata = img.load()
    w, h = img.size
    # 二值处理
    for y in range(h):
        for x in range(w):
            if pixdata[x, y] < threshold:
                pixdata[x, y] = 0
            else:
                pixdata[x, y] = 255
    return img

def depoint(img):
    # 传入二值化后的图片进行降噪
    pixdata = img.load()
    w,h = img.size
    num = 245
    for y in range(1,h-1):
        for x in range(1,w-1):
            count = 0
            if pixdata[x,y-1] > num: # 上
                count = count + 1
            if pixdata[x,y+1] > num: # 下
                count = count + 1
            if pixdata[x-1,y] > num: # 左
                count = count + 1
            if pixdata[x+1,y] > num: # 右
                count = count + 1
            if pixdata[x-1,y-1] > num: # 左上
                count = count + 1
            if pixdata[x-1,y+1] > num: # 左下
                count = count + 1
            if pixdata[x+1,y-1] > num: # 右上
                count = count + 1
            if pixdata[x+1,y+1] > num: # 右下
                count = count + 1
            if count > 4:
                pixdata[x,y] = 255
    return img

def extract():
    path = os.path.join(SAMPLES_DIR, "*")
    captcha_image_files = glob.glob(path)
    counts = {}

    for (i, captcha_image_file) in enumerate(captcha_image_files):
        filename = os.path.basename(captcha_image_file)
        if len(filename) != 8:
            continue

        captcha_correct_text = os.path.splitext(filename)[0]
        if len(captcha_correct_text) != CAPTCHA_CHARS_LEN:
            continue

        im = Image.open(captcha_image_file)
        # 灰度以及二值化
        im = binarizing(im, 190)
        # 降噪
        im = depoint(im)

        pixels = np.array(im)
        # 侦测轮廓
        # cv2.RETR_TREE
        # cv2.RETR_EXTERNAL
        contours = cv2.findContours(pixels.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, None)
        # Note: 确保你的 OpenCV 版本大于 4.0
        contours = contours[1] if imutils.is_cv3() else contours[0]

        cnts = sorted([(c, cv2.boundingRect(c)[0]) for c in contours], key=lambda x: x[1])
        arr = []
        for (c, _) in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            # if w>=9 and w<=13 and h>=15 and h<=19:
            if w>=5 and w<=25 and h>=10 and h<=25:
                Flag = False
                for i in range(-2, 3):
                    for j in range(-2, 3):
                        for k in range(-2, 3):
                            for l in range(-2, 3):
                                if (x+i, y+j, w+k, h+l) in arr:
                                    Flag = True
                                    break
                            if Flag:
                                break
                        if Flag:
                            break
                    if Flag:
                        break
                if not Flag:
                    arr.append((x, y, w, h))

        if len(arr) != CAPTCHA_CHARS_LEN:
            # logger.error("无法完成字符切割: %s", filename)
            continue

        logger.info("Extract Captcha Image: %s", filename)

        for letter_text, letter_box in zip(captcha_correct_text, arr):
            (x, y, w, h) = letter_box
            letter_image = pixels[y:y+h, x:x+w]

            save_path = os.path.join(EXTRACTED_DIR, letter_text)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            count = counts.get(letter_text, 1)
            p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
            
            logger.debug("Write Captcha Letter(%s) Image: %s", letter_text, os.path.basename(p))

            cv2.imwrite(p, letter_image)

            counts[letter_text] = count + 1


def train():
    data = []
    labels = []

    for image_file in imutils.paths.list_images(EXTRACTED_DIR):
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = resize_to_fit(image, 20, 20)
        image = np.expand_dims(image, axis=2)
        label = image_file.split(os.path.sep)[-2]

        data.append(image)
        labels.append(label)

    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    (X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.25, random_state=0)

    lb = LabelBinarizer().fit(Y_train)
    Y_train = lb.transform(Y_train)
    Y_test = lb.transform(Y_test)

    with open(MODEL_LABELS_FILENAME, "wb") as f:
        pickle.dump(lb, f)

    model = Sequential()
    model.add(Conv2D(20, (5, 5), padding="same", input_shape=(20, 20, 1), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(500, activation="relu"))
    model.add(Dense(32, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=32, epochs=10, verbose=1)
    model.save(MODEL_FILENAME)


def solve(filepath):
    with open(MODEL_LABELS_FILENAME, "rb") as f:
        lb = pickle.load(f)

    model = load_model(MODEL_FILENAME)
    image = cv2.imread(filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.copyMakeBorder(image, 20, 20, 20, 20, cv2.BORDER_REPLICATE)
    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Hack for compatibility with different OpenCV versions
    # contours = contours[0] if imutils.is_cv2() else contours[1]
    contours = contours[1] if imutils.is_cv3() else contours[0]

    letter_image_regions = []
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if w / h > 1.25:
            half_width = int(w / 2)
            letter_image_regions.append((x, y, half_width, h))
            letter_image_regions.append((x + half_width, y, half_width, h))
        else:
            letter_image_regions.append((x, y, w, h))

    if len(letter_image_regions) != 4:
        return None

    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])
    output = cv2.merge([image] * 3)
    predictions = []

    for letter_bounding_box in letter_image_regions:
        x, y, w, h = letter_bounding_box
        letter_image = image[y - 2:y + h + 2, x - 2:x + w + 2]
        letter_image = resize_to_fit(letter_image, 20, 20)
        letter_image = np.expand_dims(letter_image, axis=2)
        letter_image = np.expand_dims(letter_image, axis=0)
        prediction   = model.predict(letter_image)
        letter       = lb.inverse_transform(prediction)[0]
        predictions.append(letter)

        cv2.rectangle(output, (x - 2, y - 2), (x + w + 4, y + h + 4), (0, 255, 0), 1)
        cv2.putText(output, letter, (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)


    captcha_text = "".join(predictions)
    # print("CAPTCHA {} text is: {}".format(filepath, captcha_text))
    # cv2.imshow("Output", output)
    # cv2.waitKey(0)
    return captcha_text


def main():
    if not os.path.exists(SAMPLES_DIR):
        os.makedirs(SAMPLES_DIR)
    if not os.path.exists(EXTRACTED_DIR):
        os.makedirs(EXTRACTED_DIR)

    # extract()
    # train()
    # solve()

if __name__ == '__main__':
    logging.basicConfig(
        format  = '%(asctime)s %(levelname)-5s %(threadName)-10s %(name)-15s %(message)s',
        datefmt = '%Y-%m-%d %H:%M:%S',
        level   = logging.DEBUG,
    )

    logging.getLogger("urllib").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("tensorflow").setLevel(logging.WARNING)
    logging.getLogger("keras").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    signal.signal(signal.SIGINT,  signal.SIG_DFL)
    signal.signal(signal.SIGSEGV, signal.SIG_DFL)
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)

    main()