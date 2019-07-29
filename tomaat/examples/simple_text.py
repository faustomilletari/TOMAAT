import sys, os
# We need to add "tomaat"-directory (../..) to PATH to import the tomaat package
sys.path.append( os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))

from tomaat.server import TomaatService, TomaatApp

import SimpleITK as sitk
import numpy as np
import vtk

config = {
    "name": "Example TOMAAT demo text app",
    "modality": "Example Modality",
    "task": "Example Task",
    "anatomy": "Example Anatomy",
    "description":"Example Description",
    "port": 9000,
    "announce": False,
    "api_key": "",
}

iface_in = [{'type':'checkbox','destination':'checkbox','text':'checkbox test'}]
iface_out = [{'type':'PlainText','field':'demotext'}]

def preprocess(data_in):
    return data_in

def inference(data):
    return data

def postprocess(output):
    return {
        'demotext':["This is a demo text..."],
    }


my_app = TomaatApp(preprocess,inference,postprocess)

my_service = TomaatService(config, my_app, iface_in, iface_out)

my_service.run()
