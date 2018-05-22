import json
import requests
import tempfile
import uuid
import os
import SimpleITK as sitk
import base64
import vtk
import traceback

from urllib2 import urlopen
from klein import Klein
from twisted.internet.defer import inlineCallbacks, returnValue, DeferredLock
from twisted.internet import threads
from twisted.internet.task import LoopingCall
from twisted.internet import reactor
from twisted.logger import Logger

VERSION = 'v0.1'
ANNOUNCEMENT_SERVER_URL = 'http://tomaat.cloud:8001/announce'
ANNOUNCEMENT_INTERVAL = 1600  # seconds

OWN_PORT = 9000

logger = Logger()


def do_announcement(announcement_server_url, message):
    json_message = json.dumps(message)

    try:
        response = requests.post(announcement_server_url, data=json_message)

        response_json = response.json()

        if response_json['status'] != 0:
            logger.error('status {}'.format(response_json['status']))
            logger.error('errors: {}'.format(response_json['error']))
    except:
        logger.error('WARNING: ERROR while connecting to announcement service.')
        pass


class TomaatApp(object):
    """
    A TomaatApp is an object that implements the functionality of the user application. More specifically,
    this object implements the workflow that is needed by the app. It requires the following arguments
    """
    def __init__(self, preprocess_fun, inference_fun, postprocess_fun):
        """
        To instantiate a TomaatApp the following arguments are needed
        :type preprocess_fun: Callable function or callable object implementing pre-processing
        :type inference_fun: Callable function or callable object implementing inference
        :type postprocess_fun: Callable function or callable object implementing post-processing
        """
        super(TomaatApp, self).__init__()
        self.preprocess_fun = preprocess_fun
        self.inference_fun = inference_fun
        self.postprocess_fun = postprocess_fun

    def __call__(self, data, gpu_lock=None):
        """
        When a TomaatApp object is called it performs pre-processing, inference and post-processing
        :type data: dict dictionary containing data. The dictionary must contain the fields expected by pre-processing
        :type gpu_lock: DeferredLock optional lock to allow threads to safely use the GPU. No GPU => no lock needed
        :return: dict containing inference results after post-processing
        """
        transformed_data = self.preprocess_fun(data)

        if gpu_lock is not None:
            gpu_lock.acquire()  # acquire GPU lock

        result = self.inference_fun(transformed_data)  # GPU call

        if gpu_lock is not None:
            gpu_lock.release()  # release GPU lock

        transformed_result = self.postprocess_fun(result)

        return transformed_result


class TomaatService(object):
    klein_app = Klein()

    announcement_task = None

    gpu_lock = DeferredLock()

    def __init__(self, config, app, input_interface, output_interface):
        """
        To instantitate a TomaatService the following arguments are needed
        :type config: dict dictionary containing configuration for the service
        :type app: TomaatApp implementing the user application to be served through TomaatService
        :type input_interface: dict containing the specification for input interface (See documentation or example)
        :type output_interface: dict containing the specification for output interface (See documentation or example)
        """
        super(TomaatService, self).__init__()

        self.config = config
        self.app = app

        self.input_interface = input_interface
        self.output_interface = output_interface

    @klein_app.route('/interface', methods=['GET'])
    def interface(self, _):
        return json.dumps(self.input_interface)

    @klein_app.route('/predict', methods=['POST'])
    @inlineCallbacks
    def predict(self, request):
        logger.info('predicting...')

        result = yield threads.deferToThread(self.received_data_handler, request)

        returnValue(result)

    def start_service_announcement(
            self,
            fun=do_announcement,
            announcement_server_url=ANNOUNCEMENT_SERVER_URL,
            delay=ANNOUNCEMENT_INTERVAL
    ):
        try:
            api_key = self.config['api_key']
        except KeyError:
            raise ValueError('Api-key is missing')

        try:
            host = self.config['host']
        except KeyError:
            ip = urlopen('http://ip.42.pl/raw').read()
            port = self.config['port']
            host = 'http://' + str(ip) + ':' + str(port) + '/'
            pass

        message = {
            'api_key': api_key,
            'prediction_url': host+'/predict',
            'interface_url': host+'/interface',
            'name': self.config['name'],
            'modality': self.config['modality'],
            'task': self.config['task'],
            'anatomy': self.config['anatomy'],
            'description': self.config['description'],
        }

        self.announcement_task = LoopingCall(fun, *(announcement_server_url, message))

        self.announcement_task.start(delay)

    def stop_service_announcement(self):
        self.announcement_task.stop()

    def make_error_response(self, message):
        """
        Create simple error message to be returned to the client as plain text
        :type message: str error message to be returned to the client
        :return: response to be returned to client
        """
        response = {'type': 'PlainText', 'content': message, 'label': 'Error!'}
        return json.dumps(response)

    def parse_request(self, request):
        """
        This function takes in the content of the client message and creates a dictionary containing data.
        The service interface, that was specified in the input_interface dictionary specified at init,
        contains the specifications of the data that is needed to run this service and the fields of the dictionary
        returned by this function where the client data should be stored.
        :type request: dict request sent by the client
        :return: dict containing data that can be fed to the pre-processing, inference, post-processing pipeline
        """

        savepath = tempfile.gettempdir()

        data = {}

        for element in self.input_interface:
            raw = request.args[element['destination']]

            if element['type'] == 'volume':
                uid = uuid.uuid4()

                mha_file = str(uid) + '.mha'

                tmp_filename_mha = os.path.join(savepath, mha_file)

                with open(tmp_filename_mha, 'wb') as f:
                    f.write(raw[0])

                data[element['destination']] = [tmp_filename_mha]

            elif element['type'] == 'slider':
                data[element['destination']] = [float(raw[0])]

            elif element['type'] == 'checkbox':
                data[element['destination']] = [str(raw[0])]

            elif element['type'] == 'radiobutton':
                data[element['destination']] = [str(raw[0])]

        return data

    def make_response(self, data):
        """
        This function takes in the post-processed results of inference and creates a message for the client.
        The message is created according to the directives specified in the output_interface dictionary passed
        during instantiation of TomaatService object.
        :type request: dict containing the inference results (stored in the appropriate fields)
        :return: JSON containing response that can be returned to the client
        """
        message = []

        savepath = tempfile.gettempdir()

        for element in self.output_interface:
            type = element['type']
            field = element['field']

            if type == 'LabelVolume':
                uid = uuid.uuid4()

                mha_seg = str(uid) + '_seg.mha'
                tmp_label_volume = os.path.join(savepath, mha_seg)

                writer = sitk.ImageFileWriter()
                writer.SetFileName(tmp_label_volume)
                writer.SetUseCompression(True)
                writer.Execute(data[field][0])

                with open(tmp_label_volume, 'rb') as f:
                    vol_string = base64.encodestring(f.read())

                message.append({'type': 'LabelVolume', 'content': vol_string, 'label': ''})

                os.remove(tmp_label_volume)

            elif type == 'VTKMesh':
                uid = uuid.uuid4()

                vtk_mesh = str(uid) + '_seg.vtk'
                tmp_vtk_mesh = os.path.join(savepath, vtk_mesh)

                writer = vtk.vtkPolyDataWriter()
                writer.SetFileName(tmp_vtk_mesh)
                writer.SetInput(data[field][0])
                writer.SetFileTypeToASCII()
                writer.Write()

                with open(tmp_vtk_mesh, 'rb') as f:
                    mesh_string = base64.encodestring(f.read())

                message.append({'type': 'VTKMesh', 'content': mesh_string, 'label': ''})

                os.remove(tmp_vtk_mesh)

            elif type == 'PlainText':
                message.append({'type': 'PlainText', 'content': str(data[field][0]), 'label': ''})

        return json.dumps(message)

    def received_data_handler(self, request):
        try:
            data = self.parse_request(request)
        except:
            traceback.print_exc()
            logger.error('Server-side ERROR during request parsing')
            return self.make_error_response('Server-side ERROR during request parsing')

        try:
            transformed_result = self.app(data, gpu_lock=self.gpu_lock)
        except:
            traceback.print_exc()
            logger.error('Server-side ERROR during processing')
            return self.make_error_response('Server-side ERROR during processing')

        try:
            response = self.make_response(transformed_result)
        except:
            traceback.print_exc()
            logger.error('Server-side ERROR during response message creation')
            return self.make_error_response('Server-side ERROR during response message creation')

        return response

    def run(self):
        self.klein_app.run(port=self.config['port'], host='0.0.0.0')
        reactor.run()


