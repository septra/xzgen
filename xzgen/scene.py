from .imageconfig import Dimension, ImageData
from .imageobject import ImageObject
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import random
import cv2
import csv
import logging
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

logger = logging.getLogger(__name__)

class Scene:
    def __init__(
        self,
        dimension,
        image_data,
        scene_idx,
        place_prob,
        upper_limit=45):
        logger.debug('Initialising Scene.')

        self.upper_limit = upper_limit
        self.dimension = dimension
        self.image_data = image_data
        self.save_path = str(self.image_data.folder_name.joinpath(
            'train',
            f'multiOvlp{scene_idx}.jpg'))

        logger.debug(f'Save path for scene: {self.save_path}')
        self.place_prob = place_prob

        self.col_shelf = random.randint(2,8)
        if self.upper_limit % self.col_shelf == 0:
            self.col_shelf += 1
        self.row_shelf = self.upper_limit % self.col_shelf

        logger.debug(f'self.row_shelf:  {self.row_shelf}')
        logger.debug(f'self.col_shelf:  {self.col_shelf}')

        self.total_obj = self.row_shelf * self.col_shelf
        logger.debug(f'total_obj: {self.total_obj}')

        if self.total_obj > 4 :
            self.pos_obj = int(self.total_obj*random.uniform(0.5, 0.7))
        else:
            self.pos_obj = 1

        self.neg_obj = self.total_obj - self.pos_obj

        logger.debug(
            f'Total objects = {self.total_obj},' 
            f'Positive objects: {self.pos_obj},'
            f'Negative objects: {self.neg_obj}')

        self.obj_list = []
        self.pos_obj_list = []
        self.shelf_height = []
        self.bblist = []
        self.curr_sku_ratio = []
        self.max_real_sku_height_ratio = 0.0

        bg_height = self.dimension.bg_height
        bg_width = self.dimension.bg_width

        self.bg1 = self.get_resized_image(
            str(image_data.path_bg.joinpath(image_data.dirs_bg[
                random.randint(0, len(image_data.dirs_bg)-1)])),
            bg_height,
            bg_width)

        self.bg2 = self.get_resized_image(
            str(image_data.path_bg.joinpath(image_data.dirs_bg[
                random.randint(0, len(image_data.dirs_bg)-1)])),
            bg_height,
            bg_width)

        self.build_obj_lists(self.dimension, self.image_data)
        logger.info(f'Placing {len(self.obj_list)} objects on Scene {scene_idx}')

        self.current_col = 0
        self.current_row = 0
        self.obj_height = (bg_height-(bg_height%50))//self.row_shelf
        self.obj_width = (bg_width-(bg_width%50))//self.col_shelf
        self.x1, self.y1 = int(bg_width*0.05), int(bg_height*0.05)
        self.resize_ratio = []
        self.obj_ratio = self.obj_width/self.obj_height

        self.shelf_gaps = []
        self.cum_shelf_height = 0
        self.rf = 0.0
        self.rf2 = 0
        self.height_select = 1 
        self.max_height=0.0
        self.max_width = 0.0
        self.factor = 0.0
        self.factor1 = 0.0
        self.factor2 = 0.0

        self.build_factors()
        self.get_shelf_height()
        self.build_scene()

    def find_factors_parallel(self, position):
        logger.debug('Calling find_factors_parallel.')

        src = self.obj_list[position[0] * self.col_shelf + position[1]][0]
        logger.debug(f'Object has shape {src.shape}')
        boundRect, mask = src.find_mask()

        height = boundRect[3]
        width = boundRect[2]

        factor1 = (self.obj_height-self.shelf_gaps[position[0]])/boundRect[3]
        factor2 = self.obj_width/boundRect[2]
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
        logger.debug('Called build_factors.')
        self.shelf_gaps = [
            random.randint(
                int(0.035*self.dimension.bg_width),
                int(0.120*self.dimension.bg_width))
            for _ in range(0, self.row_shelf)]

        positions = [(i, j)
            for i in range(0, self.row_shelf)
            for j in range(0, self.col_shelf)]

        logger.debug(f'positions: {positions}')
        logger.debug(f'total positions: {len(positions)}')

        # Parallelisation
        # with ProcessPoolExecutor() as executor:
        #     results = list(
        #         executor.map(self.find_factors_parallel, positions))
        # with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        # with multiprocessing.Pool(1) as pool:
        #     results = pool.map(self.find_factors_parallel, positions)

        results = []
        for ix, position in enumerate(positions):
            logger.debug(f'Finding factors for ix: {ix} and position {position}.')
            results.append(self.find_factors_parallel(position))

        for result in results:
            if ((result['height'] > self.max_height) and
                (result['width'] > self.max_width) and
                (abs(result['shr'] - self.max_real_sku_height_ratio) < 0.001)):
                self.max_height = result['height']
                self.max_width = result['width']
                self.factor = result['factor1']
                self.factor1 = result['factor1']
                self.factor2 = result['factor2']
                self.rf = result['rf']

    def get_shelf_height(self):
        for i in range(0, self.row_shelf):
            sh_height = round(self.max_height * self.factor)
            self.cum_shelf_height += sh_height + self.shelf_gaps[i]
            if self.cum_shelf_height < self.dimension.bg_height:
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
        logger.debug('Building up object lists.')
        for i in range(0, self.pos_obj):
            idx = random.randint(0,len(image_data.pos_files_list)-1)

            self.obj_list.append([
                image_data.pos_files_list[idx]['img'],
                image_data.pos_files_list[idx]['index']])

            self.pos_obj_list.append(image_data.pos_files_list[idx])

            sku_height_ratio = (
                dimension.sku_height_ratio[image_data.pos_files_list[idx]['index']])

            if (sku_height_ratio > self.max_real_sku_height_ratio):
                self.max_real_sku_height_ratio = sku_height_ratio

        # Extends the object list for negative images.
        self.augment_negatives(image_data)

    @staticmethod
    def augment_neg_image_(img):
        logger.debug(f'augment_neg_image_ - input_shape: {img.shape}')
        aug_img = img.augment_negative()
        logger.debug(f'augment_neg_image_ - shape: {aug_img.shape}')
        return aug_img

    def augment_negatives(self, image_data):
        """Parallely augment negative images."""
        logger.debug('Calling augment_negatives.')

        random_neg_ixs = [
            random.randint(0, len(image_data.neg_files_list) - 1)
            for i in range(0, self.neg_obj)]

        neg_images_to_augment = [
            image_data.neg_files_list[ix] for ix in random_neg_ixs]
        
        # Parallelisation
        # with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        #     logger.debug('Calling multiprocessing executor for augmenting negative images.')
        #     neg_aug_images = pool.map(
        #         self.augment_neg_image_,
        #         neg_images_to_augment)

        neg_aug_images = []
        for image in neg_images_to_augment:
            neg_aug_images.append(self.augment_neg_image_(image))

        self.neg_aug_images = [[image, -1] for image in neg_aug_images]
        self.obj_list.extend(self.neg_aug_images)

    def get_resized_single_object(self, occlusion=True):
        """ Picks a random object from the object list and adds occlusion to
        it.
        """
        single_obj = self.obj_list[random.randint(0, len(self.obj_list)-1)]
        src = single_obj[0]

        boundRect, mask = src.find_mask()

        if single_obj[1] >= 0:
            rf2 = ((self.max_height / boundRect[3]) * 
            (self.dimension.sku_height_ratio[single_obj[1]] / self.max_real_sku_height_ratio))
        else:
            rf2 = (
                (self.max_height / boundRect[3]) * 
                (random.uniform(0.4, self.max_real_sku_height_ratio) / 
                self.max_real_sku_height_ratio))

        src_ = ImageObject(cv2.resize(
            src,
            None,
            fx=self.rf*rf2,
            fy=self.rf*rf2,
            interpolation=cv2.INTER_CUBIC))

        if occlusion:
            dir_occ = self.image_data.dir_occ
            path_occ = self.image_data.path_occ
            num_occ = random.randint(0, len(dir_occ) - 1)
            occ_img = cv2.imread(
                str(Path(path_occ).joinpath(dir_occ[num_occ])))
            src_ = src_.add_occlusion(occ_img)
            aug_src_ = src_.augment_obj()

        boundReact_, mask_ = aug_src_.find_mask()
        index = single_obj[1]

        return (aug_src_, index), boundReact_, mask_

    def get_objects_for_row(self, dummy):
        """ Get all objects that can fit into a single row.
        The dummy parameter is just to pass this function to the parallel
        process.
        """
        x = int(self.dimension.bg_width * 0.05) # Current placed sku row length

        src_rect_masks = []
        while True:
            (src, index), boundRect, mask = self.get_resized_single_object()
            if boundRect is None:
                continue
            if (x + boundRect[2]) > self.bg1.shape[1]*0.95:
                break
            else:
                src_rect_masks.append([(src, index), boundRect, mask])
                x += boundRect[2]
        return src_rect_masks

    def build_rows(self):
        row_ixs = range(0, self.row_shelf)

        # Parallelisation
        # with ProcessPoolExecutor() as executor:
        #     row_objects = list(executor.map(self.get_objects_for_row, row_ixs))

        rows = []
        for row_ix in row_ixs:
            rows.append(self.get_objects_for_row(row_ix))

        return rows

    def build_scene(self):
        # The self.build_rows() function will do the heavy lifting computations
        # parallely
        rows = self.build_rows()

        logger.debug(f'Total {len(rows)} rows built.')

        current_row = 0
        y1 = int(self.dimension.bg_height * 0.05)
        while True:
            inter_res = y1 + int(self.shelf_height[current_row])

            if inter_res > self.bg1.shape[0]:
                # SKU crossing height
                break

            if self.shelf_height[current_row] == 0:
                logging.info('shelf_height[current_row] = 0')
                break


            # for row_objects in rows[current_row]:
            x1 = int(self.dimension.bg_width * 0.05) # Current placed sku row length
            # for (aug_img, obj_index), boundRect, mask in row_objects:
            for (aug_img, obj_index), boundRect, mask in rows[current_row]:
                a1, b1 = (boundRect[0], boundRect[1])

                a2, b2 = (
                    boundRect[0] + boundRect[2], 
                    boundRect[1] + boundRect[3])

                crop1 = self.bg1[
                    inter_res - boundRect[3] : inter_res,
                    x1 : x1 + boundRect[2]]

                crop2 = aug_img[b1:b2, a1:a2]

                if crop1.shape != crop2.shape:
                    #logging.info('crop mismatch')
                    continue

                crop_mask = mask[b1:b2, a1:a2]
                h1, w1 = crop1.shape[:2]
                h2, w2 = crop2.shape[:2]

                skip_num = random.randint(0, 10)
                if skip_num <= self.place_prob*10:
                    crop1[crop_mask == 255] = crop2[crop_mask == 255]

                if x1+boundRect[2]+boundRect[2] < self.bg1.shape[1]*0.95:
                    if random.randint(0,333) % 3 == 0:
                        overlap_pixels = -int(0.012*self.dimension.bg_width)
                    else:
                        overlap_pixels = random.randint(
                            0,
                            int(boundRect[2]*0.27)) ##overlap percent is the last value
                else:
                    overlap_pixels = 0

                if (obj_index >= 0) and (skip_num <= self.place_prob*10):
                    annots = []
                    v1 = x1
                    annots.append(v1)
                    v2 = y1+int(self.shelf_height[current_row])-boundRect[3]
                    annots.append(v2)
                    if overlap_pixels>0:
                        v3 = x1+boundRect[2]
                        annots.append(v3)
                    else:
                        v3 = x1+boundRect[2]
                        annots.append(v3)
                    v4 = y1+int(self.shelf_height[current_row])
                    annots.append(v4)
                    sku_list_ = {v:k for k,v in self.dimension.sku_list.items()}
                    annots.append(sku_list_[obj_index])
                    self.bblist.append(annots)

                if x1+boundRect[2] < self.bg1.shape[1]*0.95:
                    x1 += int(boundRect[2])-overlap_pixels

            line_thick = random.randint(
                int(0.018*self.dimension.bg_width),
                int(0.034*self.dimension.bg_width))

            y1 += int(self.shelf_height[current_row])

            line_img = self.bg2[
                y1 : y1 + line_thick,
                int(self.dimension.bg_width*0.04):int(self.dimension.bg_width*0.96)]

            if random.randint(0,50)%4==0:
                self.bg1[
                    y1 : y1 + line_thick,
                    int(self.dimension.bg_width*0.04) : int(self.dimension.bg_width*0.96)] = line_img

            y1 += self.shelf_gaps[current_row]

        self.final_scene, self.bbs1_aug, self.sku_label = self.augmentation(
                self.bg1,
                self.bblist)
        self.csvDataTrain = []
        for i in range(0, len(self.bbs1_aug)):
            for j in range(0, len(self.bbs1_aug[i].bounding_boxes)):
                after = self.bbs1_aug[i].bounding_boxes[j]
                if int(after.x1) < 0:
                  after.x1 = 0
                if int(after.y1) < 0:
                  after.y1 = 0
                if int(after.x2) > self.bg1.shape[1]:
                  after.x2 = self.bg1.shape[1]-2
                if int(after.y2) > self.bg1.shape[0]:
                  after.y2 = self.bg1.shape[0]-2
                temp = []
                temp.append(self.save_path)
                temp.append(int(after.x1))
                temp.append(int(after.y1))
                temp.append(int(after.x2))
                temp.append(int(after.y2))
                temp.append(self.sku_label[j])
                self.csvDataTrain.append(temp)

    def write_scene(self, debug=False):
        if debug:
            img = self.final_scene[0].copy()
            for row in self.csvDataTrain:
                # row = row.split(',')
                cv2.rectangle(
                    img,
                    (row[1], row[2]),
                    (row[3], row[4]),
                    (255,0,0),
                    5)
                cv2.putText(
                    img,
                    str(row[5]),
                    (row[1], row[2]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2.0,
                    (0, 0, 255),
                    10,
                    lineType=cv2.LINE_AA)
            cv2.imwrite(self.save_path, img)
        else:
            cv2.imwrite(self.save_path, self.final_scene[0])

    def write_annotations(self):
        with open(
            self.image_data.folder_name.joinpath('train_annotations.csv'),
            'a') as csvFile:
            logger.info('Writing the train_annotations to ' + str(self.image_data.folder_name))
            writer = csv.writer(csvFile)
            writer.writerows(self.csvDataTrain)

    def show_scene(self, debug=True):
        img = self.bg1.copy()
        for row in self.csvDataTrain:
            row = row.split(',')
            cv2.rectangle(img, (row[1], row[2]), (row[3], row[4]), (255,0,0), 5)
            cv2.putText(
                img,
                str(row[5]),
                (row[1], row[2]),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0,
                (0, 0, 255),
                10,
                lineType=cv.LINE_AA)

        cv2.imshow('scene', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def augmentation(self, img, rect):
        """ Global Augmentation function.
        """
        logger.debug('Calling augmentation.')

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

