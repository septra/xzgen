from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import time
import os
import glob
import logging
import numpy as np
import cv2
import csv

from .imageobject import ImageObject

logger = logging.getLogger(__name__)

class ImageData:
    def __init__(
        self,
        path_pos,
        path_neg,
        path_nosku,
        path_occ,
        path_bg,
        sku_list,
        root_dir):
        logger.info('Initialising ImageData class.')
        self.path_pos = Path(path_pos)
        self.path_neg = Path(path_neg)
        self.path_nosku = Path(path_nosku)
        self.path_occ = Path(path_occ)
        self.dir_occ = os.listdir(self.path_occ)
        self.path_bg = Path(path_bg)
        self.dirs_bg = os.listdir(self.path_bg)
        self.sku_list = sku_list
        self.root_dir = root_dir

        self.neg_paths = self.get_image_paths(str(self.path_neg))
        self.pos_paths= self.get_image_paths(str(self.path_pos))
        self.nosku_paths = self.get_image_paths(str(self.path_nosku))

        try:
            logger.debug('Building up list of negative files.')
            with ProcessPoolExecutor() as executor:
                self.neg_files_list = list(executor.map(
                    self.read_negative_image,
                    self.neg_paths))
            logger.debug(f'Total {len(self.neg_files_list)} neg_files_list images.')
        except Exception as e:
            logger.error('Something went wrong while reading neg_files')
            logger.error(e)

        try:
            logger.debug('Building up list of positive files.')
            with ProcessPoolExecutor() as executor:
                self.pos_files_list = list(executor.map(
                    partial(self.read_positive_image, self.sku_list),
                    self.pos_paths))
            logger.debug(f'Total {len(self.pos_files_list)} pos_files_list images.')
        except Exception as e:
            logger.error('Something went wrong while reading pos files.')
            logger.error(e)

        try:
            logger.debug('Building up list of sku negative files.')
            with ProcessPoolExecutor() as executor:
                self.sku_neg_list = list(executor.map(
                    self.read_skuneg_image,
                    self.nosku_paths))
            logger.debug(f'Total {len(self.sku_neg_list)} sku_neg_list images.')
        except Exception as e:
            logger.error('Something went wrong while reading null skus.')
            logger.error(e)

        self.folder_name = self.root_dir.joinpath(
            'dataset',
            'multiSizeRelativeSkuC_' + time.strftime("%Y%m%d-%H%M%S"))

        self.folder_name.mkdir(parents=True, exist_ok=True)
        logger.debug(f'Created folder {self.folder_name}')
        self.folder_name.joinpath('train').mkdir()

        logger.info(f'Writing classes.csv to {self.folder_name}')
        with open(self.folder_name.joinpath('classes.csv'), 'w') as csvFile:
            text = "\n".join([f"{lab},{ix}" for lab, ix in sku_list.items()])
            csvFile.write(text)

    def get_image_paths(self, dir):
        paths = [f
            for path, dirs, files in os.walk(dir)
            for d in dirs
            for f in glob.iglob(os.path.join(path, d, '*.*'))]
        return paths

    @staticmethod
    def read_and_resize(path):
        img = cv2.imread(path)
        if img is None:
            raise Exception('No image found.')
        img = cv2.copyMakeBorder(
            img,
            top=10,
            bottom=10,
            left=10,
            right=10,
            borderType= cv2.BORDER_CONSTANT,
            value=[0,0,0])
        
        img = cv2.resize(
            img,
            (720, 1280),
            interpolation=cv2.INTER_CUBIC)
        return img

    @staticmethod
    def read_negative_image(path):
        logger.debug(f'Reading in negative image: {path}')
        img = ImageData.read_and_resize(path)
        return ImageObject(img)

    @staticmethod
    def read_positive_image(sku_list, path):
        logger.debug(f'Reading and resizing positive image: {path}')

        img = ImageData.read_and_resize(path)
        img_sname_list = {}
        img_sname_list['img'] = ImageObject(img)
        img_sname_list['class'] = Path(path).parent.name
        img_sname_list['index'] = sku_list[img_sname_list['class']]

        return img_sname_list

    @staticmethod
    def read_skuneg_image(path):
        img = cv2.imread(path)
        if img is None:
            raise Exception('No image found.')
        return ImageObject(img)

class Dimension:
    def __init__(self, info_file_path):
        logger.info('Creating Dimension object.')
        self.info_file_path = info_file_path
        self.valid_sku = []
        self.sku_width_ratio = []
        self.sku_height_ratio = []
        self.sku_list = {}
        self.max_dim_w = 0.0
        self.max_dim_h = 0.0
        self.sku_count = 0

        self.process_file()

        self.max_dim_h_or_w = max(self.max_dim_w, self.max_dim_h)

        self.sku_width_ratio = [
            x / self.max_dim_h_or_w 
            for x in self.sku_width_ratio]

        self.sku_height_ratio = [
            x / self.max_dim_h_or_w 
            for x in self.sku_height_ratio]

        self.bg_height = 3264
        self.bg_width = 2448

    def process_file(self):
        logger.debug(f'Processing dimension file {self.info_file_path}.')
        with open(self.info_file_path) as csv_file:
            logger.info('Reading CSV file for aspect ratio')
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                self.valid_sku.append(row[0])
                self.sku_width_ratio.append(float(row[1]))
                self.sku_height_ratio.append(float(row[2]))
                self.sku_list[row[0]] = self.sku_count
                if float(row[1]) > self.max_dim_w:
                    self.max_dim_w = float(row[1])
                if float(row[2]) > self.max_dim_h:
                    self.max_dim_h = float(row[2])
                self.sku_count += 1

