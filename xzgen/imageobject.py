import numpy as np
import cv2
import cv2 as cv
import random
import logging
from imgaug import augmenters as iaa
from imgaug import parameters as iap
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

logger = logging.getLogger(__name__)

class ImageObject(np.ndarray):
    """ Generic Image Object class to hold positives, negatives and occlusions.

    Inherits from: np.ndarray. 
    Refer - https://docs.scipy.org/doc/numpy-1.13.0/user/basics.subclassing.html
    """
    logger.info('Initialising ImageObject')
    def __new__(cls, input_array, info=None):
        obj = np.asarray(input_array).view(cls)
        obj.info = info
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.info = getattr(obj, 'info', None)

    @staticmethod
    def frompath(path):
        """ Read in the provided path and construct the ImageObject.
        """
        logger.debug(f'Initialising ImageObject from path {path}')
        img = cv.imread(path)
        if img is None:
            raise IOError(f'Image {path} not loaded.')
        return ImageObject(img)

    def augment_obj(self):
        """ The new method capturing the functionality of augment_shelf_obj'
        """
        logger.debug('augment_ob called on ImageObject')
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        seq = iaa.Sequential(
        [
            # sometimes( iaa.SaltAndPepper(0.1) ),
            # # sometimes( iaa.EdgeDetect(alpha=0.1)),
            # # sometimes( iaa.GammaContrast((1.2, 1.6))),
            # sometimes( iaa.GaussianBlur((0.5,1.25))),
            # sometimes( iaa.AverageBlur(k=9)),
            # sometimes( iaa.MedianBlur(k=7)),
            sometimes( iaa.OneOf([
                iaa.MotionBlur(k=10, angle=45),
                iaa.MotionBlur(k=10, angle=90),
                iaa.MotionBlur(k=10, angle=135),
                iaa.MotionBlur(k=10, angle=180),
                iaa.MotionBlur(k=10, angle=225),
                iaa.MotionBlur(k=10, angle=270),
                iaa.MotionBlur(k=10, angle=315),
                iaa.MotionBlur(k=10, angle=360)])),
            ],
            random_order=True
        )
        image_aug = seq.augment_image(self)

        return ImageObject(image_aug)

    def augment_negative(self):
        logger.debug('augment_negative called on ImageObject')
        bgr = self.copy()
        bbox, mask = bgr.find_mask()
        im_out = np.zeros_like(bgr)

        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        seq = iaa.Sequential(
            [
            iaa.AddToHueAndSaturation((-60, -45), per_channel=True),
            sometimes( iaa.SigmoidContrast(gain=(0, 5), cutoff=(0.0, 1.0), per_channel=True)),
            iaa.SomeOf((0, None),[
            iaa.AddToHueAndSaturation((25, 45), per_channel=True),
            iaa.LinearContrast(alpha=(0.5, 1.5), per_channel=True),
            iaa.SigmoidContrast(gain=(5, 10), cutoff=(0.0, 1.0), per_channel=True),
            iaa.LogContrast(gain=0.7, per_channel=True)
            ])
            ]
        )

        aug_det = seq.to_deterministic()
        image_aug = aug_det.augment_image(bgr)
        im_out[mask == 255] = image_aug[mask == 255]
        logger.debug(f'returning negative augmented image with shape {im_out.shape}')
        return ImageObject(im_out)

    def find_mask(self):
        src = self.copy()
        logger.debug(f'Finding mask for ImageObject with shape {src.shape[0], src.shape[1]}')
        lab1 = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
        _, temp_thresh = cv2.threshold(lab1, 1, 255, cv2.THRESH_BINARY)
        h, w = src.shape[:2]
        if cv2.countNonZero(temp_thresh)>h*w*0.5 :
            temp_thresh=cv2.bitwise_not(temp_thresh)
        kernel = np.ones((9,9),np.uint8)
        temp_thresh = cv2.erode(temp_thresh, kernel, iterations=1)
        edge1 = cv2.Laplacian(temp_thresh, cv2.CV_8U, 5)
        edge1 = cv2.morphologyEx(edge1, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(
            edge1,
            cv2.RETR_CCOMP,
            cv2.CHAIN_APPROX_SIMPLE)
        boundRect = None
        maxArea = 0
        h, w = edge1.shape[:2]
        tp = h*w
        idx = -1
        mask = np.zeros_like(edge1)

        for i, cnt in enumerate(contours):
            area = cv2.contourArea( cnt )
            if area>maxArea and area<tp*0.9 :
                boundRect = cv2.boundingRect( cnt )
                maxArea = area
                idx = i
        cv2.drawContours(
            mask,
            contours,
            idx,
            (255,255,255),
            -1,
            lineType=cv2.LINE_AA)

        return boundRect, mask

    def add_occlusion(self, occ_img):
        """ Add occlusion occ_img to the image
        """
        logger.debug('Adding occlusion to ImageObject')
        logger.debug(f'Occlusion Image shape: {occ_img.shape}')
        logger.debug(f'Self shape: {self.shape} - gets assigned to sku_img')

        sku_img = self.copy()
        occ_img = ImageObject(occ_img)

        if random.randint(0,13)%6!=0:
            return sku_img

        bbox, mask_occ = occ_img.find_mask()

        im_out = np.zeros_like(occ_img)
        seq = iaa.Sequential(
            [ iaa.AddToHueAndSaturation((-45, 45), per_channel=True),])

        aug_det = seq.to_deterministic()
        image_aug = ImageObject(aug_det.augment_image(occ_img))

        logger.debug(
            'add_occlusion called. Shapes: '
            f'im_out: {im_out.shape} '
            f'mask_occ: {mask_occ.shape} '
            f'image_aug: {image_aug.shape} ')

        im_out[mask_occ == 255] = image_aug[mask_occ == 255]

        occ_img = ImageObject(im_out)
        boundRect, mask = sku_img.find_mask()
        boundRect2, mask2 = occ_img.find_mask()
        fx = boundRect[2]*0.3/boundRect2[2] #Choose occlusion size
        fy = boundRect[3]*0.3/boundRect2[3] #Choose occlusion size
        occ_img = cv2.resize(occ_img,None,fx=fx,fy=fy,interpolation=cv2.INTER_CUBIC)
        occ_img = ImageObject(occ_img)
        boundRect3, mask3 = occ_img.find_mask()
        contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        tx = ty = -1
        loop_cnt = 0
        while tx<=0 or ty<=0:  
            rnd_pt = contours[0][random.randint(0,len(contours[0])-1)]
            tx = rnd_pt[0][0]-boundRect3[2]//2
            ty = rnd_pt[0][1]-boundRect3[3]//2
            if tx + boundRect3[2] > sku_img.shape[1] or tx < 0:
                tx = -1
            if ty + boundRect3[3] > sku_img.shape[0] or ty < 0:
                ty = -1
            loop_cnt += 1
            if loop_cnt >500:
                break
        if tx<0 or ty<0:
            return sku_img

        crop1 = sku_img[ty:ty+boundRect3[3], tx:tx+boundRect3[2]]
        crop2 = occ_img[boundRect3[1]:boundRect3[1]+boundRect3[3], boundRect3[0]:boundRect3[0]+boundRect3[2]]
        crop_mask = mask3[boundRect3[1]:boundRect3[1]+boundRect3[3], boundRect3[0]:boundRect3[0]+boundRect3[2]]
        crop1[crop_mask == 255] = crop2[crop_mask == 255]
        return sku_img

