from .imageconfig import Dimension, ImageData
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import random
import cv2 as cv
from imgaug import augmenters as iaa
from imgaug import parameters as iap
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

class Scene:
    def __init__(self, dimension, image_data, shelf_count, upper_limit=45):
        self.upper_limit = upper_limit
        self.dimension = dimension
        self.save_path = str(folder_name.joinpath(
            'train',
            f'multiOvlp{shelf_count}.jpg'))

        self.col_shelf = random.randint(2,8)
        if self.upper_limit % self.col_shelf == 0:
            self.col_shelf += 1
        self.row_shelf = self.upper_limit % self.col_shelf

        self.total_obj = self.row_shelf * self.col_shelf
        if self.total_obj > 4 :
            self.pos_obj = int(self.total_obj*random.uniform(0.5, 0.7))
        else:
            self.pos_obj = 1
        self.neg_obj = self.total_obj - self.pos_obj

        self.obj_list = []
        self.pos_obj_list = []
        self.shelf_height = []
        self.bblist1 = []
        self.curr_sku_ratio = []
        self.max_real_sku_height_ratio = 0.0

        bg_height = self.dimension.bg_height
        bg_width = self.dimension.bg_width

        self.bg1 = self.get_resized_image(
            path_bg + dirs_bg[random.randint(0, len(dirs_bg)-1)],
            bg_height,
            bg_width)

        self.bg2 = self.get_resized_image(
            path_bg + dirs_bg[random.randint(0, len(dirs_bg)-1)],
            bg_height,
            bg_width)

        self.build_obj_lists()

        self.current_col = 0
        self.current_row = 0
        self.obj_height = (bg_height-(bg_height%50))//self.row_shelf
        self.obj_width = (bg_width-(bg_width%50))//self.col_shelf
        self.x1, self.y1 = int(bg_width*0.05), int(bg_height*0.05)
        self.resize_ratio = []
        self.obj_ratio = self.obj_width/self.obj_height

        self.shelf_gap = []
        self.cum_shelf_height = 0
        self.rf = 0.0
        self.rf2 = 0
        self.height_select = 1 
        self.max_height=0.0
        self.max_width = 0.0
        self.factor = 0.0
        self.factor1 = 0.0
        self.factor2 = 0.0

    def find_factors_parallel(self, position):
        src = self.obj_list[position[0] * self.col_shelf + position[1]][0]
        boundRect, mask = self.find_mask(src)

        height = boundRect[3]
        width = boundRect[2]

        factor1 = (obj_height-self.shelf_gaps[position[0]])/boundRect[3]
        factor2 = obj_width/boundRect[2]
        rf = min(factor1, factor2)

        shr = self.dimension.sku_height_ratio[
            self.obj_list[position[0] * self.col_shelf + position[1]][1]]

        return {
            'height': height, 
            'width': width, 
            'factor1': factor1, 
            'factor2': factor2, 
            'rf': rf, 
            'shr': shr}

    def build_factors(self):
        self.shelf_gaps = [
            random.randint(int(0.035*bg_width), int(0.12*bg_width))
            for _ in range(0, self.row_shelf)]

        positions = [(i, j)
            for i in range(0, self.row_shelf)
            for j in range(0, self.col_shelf)]

        with ProcessPoolExecutor() as executor:
            results = list(
                excecutor.map(self.find_factors_parallel, positions))

        for result in results:
            if ((result['height'] > self.max_height) and
                (result['width'] > self.max_width) and
                (abs(result['shr'] - self.max_real_sku_height_ratio) < 0.001)):
                self.max_height = result['height']
                self.max_width = result['width']
                self.factor = result['factor1']
                self.factor1 = result['factor1']
                self.factor2 = result['factor2']
                self.rf = result[rf]

    def get_shelf_height(self):
        for i in range(0, row_shelf):
            sh_height = round(self.max_height * self.factor)
            self.cum_shelf_height += sh_height + self.shelf_gap[i]
            if self.cum_shelf_height < self.bg_height:
                self.shelf_height.append(round(self.max_height * self.factor)-5) 
            else:
                self.shelf_height.append(0)

    @staticmethod
    def get_resized_image(path, height, width):
        image = cv2.imread(path)
        image = cv2.resize(
            image ,
            (int(width), int(height)),
            interpolation=cv2.INTER_CUBIC)
        return image

    def build_obj_lists(self, dimension: Dimension, image_data: ImageData):
        for i in range(0, self.pos_obj):
            idx = random.randint(0,len(pos_files_list)-1)

            self.obj_list.append([
                image_data.pos_files_list[idx]['class'],
                image_data.pos_files_list[idx]['index']])

            self.pos_obj_list.append(image_data.pos_files_list[idx])

            sku_height_ratio = (
                dimension.
                sku_height_ratio[image_data.pos_files_list[idx][2]])

            if (sku_height_ratio > self.max_real_sku_height_ratio):
                self.max_real_sku_height_ratio = sku_height_ratio

    def augment_negatives(self, image_data):
        """Parallely augment negative images."""
        random_neg_ixs = [
            random.randint(0, len(image_data.neg_files_list) - 1)
            for i in range(0, neg_obj)]

        neg_images_to_augment = [
            image_data.neg_files_list[ix] for ix in random_neg_ixs]

        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            neg_aug_images = pool.map(
                augment_neg_image,
                neg_images_to_augment)

        self.neg_aug_images = [[image, -1] for image in neg_aug_images]
        self.obj_list.extend(neg_aug_images)

    def augmentation(self, img, rect):
        """ Global Augmentation function.
        """
        logging.info('Calling augmentation.')

        bb_list1 = []
        bb = []
        imglist = []
        sku_label = []

        imglist.append(img)
        if type(rect) is list:
            for bbox in rect:
                bb.append(BoundingBox(
                    x1=int(bbox[0]),
                    y1=int(bbox[1]),
                    x2=int(bbox[2]),
                    y2=int(bbox[3])))
                sku_label.append(bbox[4])
            bbs1 = ia.BoundingBoxesOnImage(bb, shape=img.shape)
            bb_list1.append(bbs1)
        else:
            bb_list1.append(ia.BoundingBoxesOnImage([BoundingBox(
                x1=rect[0],
                y1=rect[1],
                x2=rect[0]+rect[2],
                y2=rect[1]+rect[3])] ,shape=img.shape))

        # ia.seed(10)

        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        seq = iaa.Sequential(
            [   
                iaa.PerspectiveTransform(
                    scale=(0, 0.15),
                    scalew=(0.0, 0.1),
                    cval=(0, 255),
                    mode='replicate',
                    keep_size=False),
                sometimes( iaa.GammaContrast((1.0, 1.2))),
                sometimes (iaa.OneOf([
                    iaa.GaussianBlur((0.5, 1.0)),
                    iaa.AverageBlur(k=5),
                    iaa.MedianBlur(k=(3, 5)),
                    ])),
                sometimes( iaa.OneOf([
                    iaa.MotionBlur(k=12, angle=90),
                    iaa.MotionBlur(k=12, angle=180),
                    iaa.MotionBlur(k=12, angle=270),
                    iaa.MotionBlur(k=12, angle=360)])),
                sometimes (iaa.OneOf([
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255)),
                    iaa.AdditiveLaplaceNoise(loc=0, scale=(0.0, 0.05*255)),
                    ])),
            ],
            random_order=True
        )

        aug_det = seq.to_deterministic()
        image_aug = aug_det.augment_images(imglist)
        bb_aug = aug_det.augment_bounding_boxes(bb_list1)

        if type(rect) is list:
            return image_aug, bb_aug, sku_label
        else:
            return image_aug, bb_aug

        def save_scene(self):
            pass

