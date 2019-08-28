import numpy as np
import cv2 as cv
from imgaug import augmenters as iaa
from imgaug import parameters as iap
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

class ImageObject(np.ndarray):
    """ Generic Image Object class to hold positives, negatives and occlusions.

    Inherits from: np.ndarray. 
    Refer - https://docs.scipy.org/doc/numpy-1.13.0/user/basics.subclassing.html
    """
    def __new__(cls, input_array, info=None):
        obj = np.asarray(input_array).view(cls)
        obj.info = info
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.info = getattr(obj, 'info', None)

    @staticmethod
    def from_path(path):
        """ Read in the provided path and construct the ImageObject.
        """
        img = cv.imread(path)
        if img is None:
            raise IOError(f'Image {path} not loaded.')
        return ImageObject(img)

    def augment_obj(self):
        """ The new method capturing the functionality of augment_image_obj'
        """
        sometimes = lambda aug: iaa.Sometimes(0.3, aug)

        seq = iaa.Sequential(
        [
            sometimes( iaa.SaltAndPepper(0.2) ),
            sometimes( iaa.SigmoidContrast(cutoff=0.25, gain=5)),
            sometimes( iaa.SigmoidContrast(cutoff=0.4, gain=5)),
            sometimes( iaa.EdgeDetect(alpha=0.1)),       
            sometimes( iaa.GammaContrast((0.7,1.4))),
            sometimes( iaa.GaussianBlur((0.5,1.25))),
            sometimes( iaa.AverageBlur(k=3)),
            sometimes( iaa.MedianBlur(k=3)),
        ],
            random_order=True
        )
        aug_det = seq.to_deterministic()
        image_aug = aug_det.augment_image(self.src)
        return ImageObject(image_aug)

    @staticmethod
    def find_mask(self):
        src = self.src.copy()
        lab1 = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
        _, temp_thresh = cv2.threshold(lab1, 1, 255, cv2.THRESH_BINARY)
        h, w = src1.shape[:2]
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
        sku_img = self.copy()
        if random.randint(0,13)%6!=0:
            return sku_img

        bbox, mask_occ = occ_img.find_mask()

        im_out = np.zeros_like(occ_img)
        seq = iaa.Sequential(
            [ iaa.AddToHueAndSaturation((-45, 45), per_channel=True),])

        aug_det = seq.to_deterministic()
        image_aug = ImageObject(aug_det.augment_image(sku_image))

        im_out[mask_occ == 255] = image_aug[0][mask_occ == 255]
        occ_img = ImageObject(im_out)
        boundRect, mask = sku_image.find_mask()
        boundRect2, mask2 = occ_img.find_mask()
        fx = boundRect[2]*0.3/boundRect2[2] #Choose occlusion size
        fy = boundRect[3]*0.3/boundRect2[3] #Choose occlusion size
        occ_img = cv2.resize(occ_img,None,fx=fx,fy=fy,interpolation=cv2.INTER_CUBIC)
        occ_img = ImageObject(occ_img)
        boundRect3, mask3 = occ_image.find_mask()
        _, contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

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

