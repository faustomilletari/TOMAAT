import sys, os
# We need to add "tomaat"-directory (../..) to PATH to import the tomaat package
sys.path.append( os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))

from tomaat.server import TomaatService, TomaatApp

import SimpleITK as sitk
import numpy as np

config = {
    "name": "Example TOMAAT fiducials negation app",
    "modality": "Example Modality",
    "task": "Example Task",
    "anatomy": "Example Anatomy",
    "description":"Example Description",
    "port": 9000,
    "announce": False,
    "api_key": "",
}

iface_in = [{'type':'fiducials','destination':'fids'}]
iface_out = [{'type':'Fiducials','field':'fids'}]

def preprocess(data_in):
    return {'fids':[data_in['fids'][0]] }

def inference(data):
    fids_np = data['fids'][0]
    fids_neg = fids_np * -1.0
    return {
        'fids_neg': fids_neg,
    }

def postprocess(output):
    fids_neg = output['fids_neg'].astype(np.float)
    return {
        'fids':[fids_neg]
    }


my_app = TomaatApp(preprocess,inference,postprocess)

my_service = TomaatService(config, my_app, iface_in, iface_out)

my_service.run()
