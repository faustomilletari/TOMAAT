import sys, os
# We need to add "tomaat"-directory (../..) to PATH to import the tomaat package
sys.path.append( os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))

from tomaat.server import TomaatService, TomaatApp

import SimpleITK as sitk
import numpy as np

config = {
    "name": "Example TOMAAT thresholding app",
    "modality": "Example Modality",
    "task": "Example Task",
    "anatomy": "Example Anatomy",
    "description":"Example Description",
    "port": 9000,
    "announce": False,
    "api_key": "",
}

iface_in = [{'type':'volume','destination':'images'}]
iface_out = [{'type':'LabelVolume','field':'images','label':'test'}]

def preprocess(input):
    return {'images':[sitk.ReadImage(input['images'][0])] }

def inference(data):
    itk_image = data['images'][0]
    image_data = sitk.GetArrayFromImage(itk_image)
    treshold = np.mean(image_data)
    image_binary = image_data >= treshold
    return {
        'image_data': image_binary,
        'image_old': itk_image,
    }

def postprocess(output):
    binarized_image = output['image_data'].astype(np.float)
    image_new = sitk.GetImageFromArray(binarized_image)
    image_new.CopyInformation(output['image_old'])
    return {
        'images':[image_new]
    }


my_app = TomaatApp(preprocess,inference,postprocess)

my_service = TomaatService(config, my_app, iface_in, iface_out)

my_service.run()
