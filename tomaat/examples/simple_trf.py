import sys, os
# We need to add "tomaat"-directory (../..) to PATH to import the tomaat package
sys.path.append( os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))

from tomaat.server import TomaatService, TomaatApp

import SimpleITK as sitk
import numpy as np

config = {
    "name": "Example TOMAAT transform direction flip app",
    "modality": "Example Modality",
    "task": "Example Task",
    "anatomy": "Example Anatomy",
    "description":"Example Description",
    "port": 9000,
    "announce": False,
    "api_key": "",
}

iface_in = [{'type':'transform','destination':'trf'}]
iface_out = [{'type':'TransformGrid','field':'trf'}]

def preprocess(data_in):
    print(data_in)
    return {'trf':[sitk.ReadImage(data_in['trf'][0])] }

def inference(data):
    itk_trf = data['trf'][0]
    trf_data = sitk.GetArrayFromImage(itk_trf)
    trf_data *= -1.0
    return {
        'trf_data': trf_data,
        'trf_old': itk_trf,
    }

def postprocess(output):
    trf_new_itk = output['trf_data'].astype(np.float)
    trf_new = sitk.GetImageFromArray(trf_new_itk)
    trf_new.CopyInformation(output['trf_old'])
    return {
        'trf':[trf_new]
    }


my_app = TomaatApp(preprocess,inference,postprocess)

my_service = TomaatService(config, my_app, iface_in, iface_out)

my_service.run()
