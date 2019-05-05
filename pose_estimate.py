import cv2
import math
import numpy as np


from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

class PoseEstimator(object):
    """
    The PoseEstimator class manages all operations related
    to Pose Estimation. It acts as a wrapper on top of
    TfPoseEstimator implement inside openpose model.
    The class supplies human pose coorinates to
    requestor objects.
    """
    resize_out_ratio = 4.0  # no of relevance, kept for sake of completeness

    def __init__(self, resize='0x0', model='mobilenet_thin'):

        self.humans = None      # list of humans with pose info
        self.image = None
        self.bboxes = []        # list of bbox [x1, y1, x2, y2]
        """
        Two available models are cmu & mobilenet_thin
        if running in CPU mode only, then mobilenet_thin
        is recommended. Default fetches mobilenet_thin
        """
        self.model = model

        """
        if resize value is provided, it will resize images
        before they are processed. default=0x0, Recommends:
        432x368 or 656x368 or 1312x736
        """
        self.w, self.h = model_wh(resize)
        self.loadModel()

    def loadModel(self):
        """
        Loads the cmu or mobilenet model in memory
        """
        try:
            if self.w == 0 or self.h == 0:
                self.e = TfPoseEstimator(get_graph_path(self.model),
                        target_size=(432, 368))
            else:
                self.e = TfPoseEstimator(get_graph_path(self.model),
                        target_size=(self.w, self.h))
        except MemoryError:
            print ("couldn't load model into memory...")


    def infer(self, image):
        """
        calls the inference API inside tf_pose (openpose)
        returning the poses of humans and drawing the skeleton
        on image frame
        """
        self.image = image
        if self.image is None:
            raise Exception('The image is not valid. check your image')

        self.humans = self.e.inference(self.image,
                resize_to_default=(self.w > 0 and self.h > 0),
                upsample_size=self.resize_out_ratio)
        self.image = TfPoseEstimator.draw_humans(self.image, self.humans,
                imgcopy=False)
        return self.image

    def getHumans(self):
        return self.humans

    def getImage(self):
        return self.image

    def _normalize_values(self, width, height):
        if self.w == 0:
            width = width * 432
        else:
            width = width * self.w
        if self.h == 0:
            height = height * 368
        else:
            height = height * self.h

        return width, height

    def getBboxes(self):
      return self.bboxes

    def getKeypoints(self):
        """
        Returns a list of keypoints of all
        the persons in a frame
        keypt_list = [keypts1, keypts2]
        keypts = [x1, y1, score, ...]
        """
        keypt_list = []
        for human in self.humans:
            keypts = []
            for key, values in human.body_parts.items():
                # print (key, 'x val %.2f' % values.x, 'y val %.2f' % values.y)
                # print ('values.part_idx, values.uidx ',
                #           values.part_idx, values.uidx)
                x, y = self._normalize_values(values.x, values.y)
                keypts.extend([x, y, values.score])
            keypt_list.append(keypts)
        #print (keypt_list)
        return keypt_list

    def _updateBboxes(self):
      self.bboxes = []
      for human in self.humans:
        min_x, min_y = math.inf, math.inf
        max_x, max_y = -1, -1
        bbox = [min_x, min_y, max_x, max_y]
        for key, values in human.body_parts.items():
          if values.x < min_x:
            min_x = values.x
          if values.y < min_y:
            min_y = values.y
          if values.x > max_x:
            max_x = values.x
          if values.y > max_y:
            max_y = values.y
        bbox = [min_x, min_y, max_x, max_y]
        self.bboxes.append(bbox)

    def showResults(self):
        """
        utility method for debug purposes,
        not called from anywhere
        """
        #print (self.humans)

        cv2.imshow('tf-pose-estimation result', self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
