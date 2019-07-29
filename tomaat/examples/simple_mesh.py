import sys, os
# We need to add "tomaat"-directory (../..) to PATH to import the tomaat package
sys.path.append( os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))

from tomaat.server import TomaatService, TomaatApp

import SimpleITK as sitk
import numpy as np
import vtk

config = {
    "name": "Example TOMAAT monkey mesh app",
    "modality": "Example Modality",
    "task": "Example Task",
    "anatomy": "Example Anatomy",
    "description":"Example Description",
    "port": 9000,
    "announce": False,
    "api_key": "",
}

iface_in = [{'type':'checkbox','destination':'checkbox','text':'checkbox test'}]
iface_out = [{'type':'VTKMesh','field':'mesh','text':'monkey'}]

def preprocess(data_in):
    monkey = "monkey.stl"
    script_dir = os.path.dirname(__file__)
    monkey_path = os.path.join(script_dir,monkey)
    return {
        "mesh_path":monkey_path
    }

def inference(data):
    r = vtk.vtkSTLReader()
    r.SetFileName(data['mesh_path'])
    r.Update()
    monkey_mesh = r.GetOutput()
    return {
        'mesh_vtk':monkey_mesh
    }

def postprocess(output):
    return {
        'mesh':[output['mesh_vtk']],
    }


my_app = TomaatApp(preprocess,inference,postprocess)

my_service = TomaatService(config, my_app, iface_in, iface_out)

my_service.run()
