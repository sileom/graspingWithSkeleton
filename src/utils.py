import os
#from pathlib import Path
import cv2 as cv

class Utils:

    @staticmethod
    def root():
        #return Path(__file__).parent.parent
        return os.path.abspath(os.path.join(__file__, "../../"))
    
    @staticmethod
    def absolute_path(rel_path):
        return os.path.join(Utils.root(), rel_path.replace('/', os.sep))

    @staticmethod
    def load_classes(path):
        labels = []
        with open(Utils.absolute_path(path), "r") as f:
            labels = [cname.strip() for cname in f.readlines()]
        return labels

    @staticmethod
    def is_cuda_cv(): 
        try:
            count = cv.cuda.getCudaEnabledDeviceCount()
            if count > 0:
                return 1
            else:
                return 0
        except:
            return 0
    
    @staticmethod
    def get_detection_class(class_id):
        print(class_id)
        #return "#123456" #classi[class_id]
        return classi[class_id]


'''python 3
class Utils:

    @staticmethod
    def root():
        return Path(__file__).parent.parent
    
    @staticmethod
    def absolute_path(rel_path:str) -> str:
        return os.path.join(Utils.root(), rel_path.replace('/', os.sep))

    @staticmethod
    def load_classes(path:str) -> list:
        labels = []
        with open(Utils.absolute_path(path), "r") as f:
            labels = [cname.strip() for cname in f.readlines()]
        return labels

    @staticmethod
    def is_cuda_cv(): 
        try:
            count = cv.cuda.getCudaEnabledDeviceCount()
            if count > 0:
                return 1
            else:
                return 0
        except:
            return 0
'''
