import json
import requests
import tempfile
import uuid
import os
import SimpleITK as sitk
import base64
import traceback
import numpy as np
import shutil

try:
    # For Python 3.0 and later
    from urllib.request import urlopen
except ImportError:
    # Fall back to Python 2's urllib2
    from urllib2 import urlopen

from multiprocessing import Process, Manager, Lock

from klein import Klein
from twisted.internet.defer import inlineCallbacks, returnValue, DeferredLock
from twisted.internet import threads
from twisted.internet.task import LoopingCall
from twisted.internet import reactor
from twisted.logger import Logger


ANNOUNCEMENT_SERVER_URL = 'http://tomaat.cloud:8001/announce'
ANNOUNCEMENT_INTERVAL = 1600  # seconds

logger = Logger()


def is_base64(s):
    try:
        if base64.b64encode(base64.b64decode(s)) == s:
            return True
    except:
        pass
    return False


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
        if not "cert_path" in self.config.keys():
            self.config["cert_path"] = "./tomaat_cert"

        cert_private = self.config["cert_path"] + ".key"
        cert_public = self.config["cert_path"] + ".crt"

        from . import makecert
        if not os.path.exists(cert_private) or not os.path.exists(cert_public):
            makecert.create_self_signed_cert(cert_public,cert_private)

        cert_fingerprint = makecert.get_cert_fingerprint(cert_public)
        print("\nMake sure to check the fingerprint of this endpoint on the client side.\nThe fingerprint is:\n\n{}\n".format(cert_fingerprint))

        # setup https
        endpoint_specification = "ssl:{}".format(self.config['port'])
        endpoint_specification += ":certKey="+cert_public
        endpoint_specification += ":privateKey="+cert_private

        self.config['endpoint_specification'] = endpoint_specification


    @klein_app.route('/announcePoint', methods=['GET'])
    def announcePoint(self, request):
        try: ap = self.config['announcement']
        except: ap = ""
        return json.dumps({"announced_at":ap})

    @klein_app.route('/interface', methods=['GET'])
    def interface(self, request):
        request.setHeader('Access-Control-Allow-Origin', '*')
        request.setHeader('Access-Control-Allow-Methods', 'GET')
        request.setHeader('Access-Control-Allow-Headers', '*')
        request.setHeader('Access-Control-Max-Age', '2520')  # 42 hours

        return json.dumps(self.input_interface)

    @klein_app.route('/predict', methods=['POST'])
    @inlineCallbacks
    def predict(self, request):
        request.setHeader('Access-Control-Allow-Origin', '*')
        request.setHeader('Access-Control-Allow-Methods', 'POST')
        request.setHeader('Access-Control-Allow-Headers', '*')
        request.setHeader('Access-Control-Max-Age', '520')  # 42 hours

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
        response = [{'type': 'PlainText', 'content': message, 'label': 'Error!'}]
        return response

    def parse_request(self, request, savepath):
        """
        This function takes in the content of the client message and creates a dictionary containing data.
        The service interface, that was specified in the input_interface dictionary specified at init,
        contains the specifications of the data that is needed to run this service and the fields of the dictionary
        returned by this function where the client data should be stored.
        :type request: dict request sent by the client
        :return: dict containing data that can be fed to the pre-processing, inference, post-processing pipeline
        """

        data = {}

        for element in self.input_interface:
            raw = request.args[element['destination'].encode('UTF-8')]

            if element['type'] == 'volume':
                uid = uuid.uuid4()

                mha_file = str(uid).replace('-', '') + '.mha'

                tmp_filename_mha = os.path.join(savepath, mha_file)

                with open(tmp_filename_mha, 'wb') as f:
                    if is_base64(raw[0]):
                        f.write(base64.decodestring(raw[0]))
                    else:
                        f.write(raw[0])
                        print(
                            'Your client has passed RAW file content instead of base64 encoded string: '
                            'this is deprecated and will result in errors in future version of the server'
                        )
                data[element['destination']] = [tmp_filename_mha]

            elif element['type'] == 'slider':
                data[element['destination']] = [float(raw[0])]

            elif element['type'] == 'checkbox':
                data[element['destination']] = [str(raw[0])]

            elif element['type'] == 'radiobutton':
                data[element['destination']] = [str(raw[0])]

            elif element['type'] == 'fiducials':
                # Each coordinate is separated by ';'
                # Each coord value is separated by ','
                fiducial_string = str(raw[0])
                fiducial_list = [ [ float(val) for val in coords.split(',')] for coords in fiducial_string.split(';')]
                data[element['destination']] = [np.asarray(fiducial_list)]

            elif element['type'] == 'transform':
                dtype = {
                  'nii.gz':'grid',
                  'h5': 'bspline',
                  'mat':'linear'
                }
                # transform encoding:
                # <filetype> newline
                # <base64 of file>

                # determine file type
                req = str(raw[0])
                trf_file_type = ""
                for trf_type in dtype.keys():
                    if req.startswith(trf_type+"\n"):
                        trf_file_type = '.' + trf_type

                if not trf_file_type:
                    # invalid format
                    return

                # store file
                uid = uuid.uuid4()
                trf_file = str(uid) + trf_file_type
                tmp_transform = os.path.join(savepath, trf_file)
                with open(tmp_transform, 'wb') as f:
                    # write base64 data
                    f.write(base64.decodestring(req[len(trf_file_type):]))

                data[element['destination']] = [tmp_transform]

        return data

    def make_response(self, data, savepath):
        """
        This function takes in the post-processed results of inference and creates a message for the client.
        The message is created according to the directives specified in the output_interface dictionary passed
        during instantiation of TomaatService object.
        :type request: dict containing the inference results (stored in the appropriate fields)
        :return: JSON containing response that can be returned to the client
        """
        message = []

        for element in self.output_interface:
            type = element['type']
            field = element['field']

            if type == 'LabelVolume':
                uid = uuid.uuid4()

                mha_seg = str(uid).replace('-', '') + '_seg.mha'
                tmp_label_volume = os.path.join(savepath, mha_seg)

                writer = sitk.ImageFileWriter()
                writer.SetFileName(tmp_label_volume)
                writer.SetUseCompression(True)
                writer.Execute(data[field][0])

                with open(tmp_label_volume, 'rb') as f:
                    vol_string = base64.encodestring(f.read()).decode('utf-8')

                message.append({'type': 'LabelVolume', 'content': vol_string, 'label': ''})

                os.remove(tmp_label_volume)

            elif type == 'VTKMesh':
                import vtk

                uid = uuid.uuid4()

                vtk_mesh = str(uid).replace('-', '') + '_seg.vtk'
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

            elif type == 'Fiducials':
                fiducial_array = data[field][0]
                fiducial_str = ';'.join([','.join(map(str,fid_point)) for fid_point in fiducial_array])
                message.append({'type': 'Fiducials', 'content': fiducial_str, 'label': ''})

            elif type in ['TransformGrid','TransformBSpline','TransformLinear']:
                trf_file_type = {
                    'TransformGrid':'nii.gz',
                    'TransformBSpline':'h5',
                    'TransformLinear':'mat',
                }
                uid = uuid.uuid4()

                trf_file_name = str(uid) + '.' + trf_file_type[type]
                trf_file_path = os.path.join(savepath, trf_file_name)

                if type == "TransformGrid":
                    # Displacement fields are stored as regular volumes.
                    sitk.WriteImage(data[field][0],trf_file_path)
                else:
                    sitk.WriteTransform(data[field][0],trf_file_path)

                with open(trf_file_path, 'rb') as f:
                    vol_string = base64.encodestring(f.read()).decode("utf-8")

                message.append({'type': type, 'content': vol_string, 'label': ''})

                os.remove(trf_file_path)

        return message

    def received_data_handler(self, request):
        savepath = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()).replace('-', ''))

        os.mkdir(savepath)

        try:
            data = self.parse_request(request, savepath)
        except:
            traceback.print_exc()
            logger.error('Server-side ERROR during request parsing')
            response = self.make_error_response('Server-side ERROR during request parsing')
            return json.dumps(response)

        try:
            transformed_result = self.app(data, gpu_lock=self.gpu_lock)
        except:
            traceback.print_exc()
            logger.error('Server-side ERROR during processing')
            response = self.make_error_response('Server-side ERROR during processing')
            return json.dumps(response)

        try:
            response = self.make_response(transformed_result, savepath)
        except:
            traceback.print_exc()
            logger.error('Server-side ERROR during response message creation')
            response = self.make_error_response('Server-side ERROR during response message creation')
            return json.dumps(response)

        shutil.rmtree(savepath)
        
        return json.dumps(response)

    def run(self):
        endpoint_specification = self.config.get("endpoint_specification",None)

        self.klein_app.run(port=self.config['port'], host='0.0.0.0', endpoint_description=endpoint_specification)
        reactor.run()


class TomaatServiceDelayedResponse(TomaatService):
    announcement_task = None

    gpu_lock = DeferredLock()
    
    multiprocess_manager = Manager()

    result_dict = multiprocess_manager.dict()
    reqest_list = multiprocess_manager.list()

    multiprocess_lock = Lock()

    klein_app = Klein()

    def __init__(self, no_concurrent_thread_execution=True, **kwargs):
        super(TomaatServiceDelayedResponse, self).__init__(**kwargs)
        self.no_concurrent_thread_execution = no_concurrent_thread_execution

    def received_data_handler(self, request):
        req_id = str(uuid.uuid4()).replace('-', '')

        savepath = os.path.join(tempfile.gettempdir(), req_id)

        os.mkdir(savepath)

        def processing_thread():
            if self.no_concurrent_thread_execution:
                self.multiprocess_lock.acquire()

            response = self.make_error_response('Server-side ERROR during processing')

            try:
                data = self.parse_request(request, savepath)
            except:
                traceback.print_exc()
                logger.error('Server-side ERROR during request parsing')

            try:
                transformed_result = self.app(data, gpu_lock=self.gpu_lock)
            except:
                traceback.print_exc()
                logger.error('Server-side ERROR during processing')

            try:
                response = self.make_response(transformed_result, savepath)
            except:
                traceback.print_exc()
                logger.error('Server-side ERROR during response message creation')

            response = [{
                'type': 'PlainText',
                'content': 'The results of your earlier request {} have been received'.format(req_id),
                'label': ''
            }] + response

            self.result_dict[req_id] = response

            shutil.rmtree(savepath)

            if self.no_concurrent_thread_execution:
                self.multiprocess_lock.release()

        delegated_process = Process(target=processing_thread, args=())
        delegated_process.start()

        self.reqest_list.append(req_id)

        response = [{'type': 'DelayedResponse', 'request_id': req_id}]

        return json.dumps(response)

    @klein_app.route('/interface', methods=['GET'])
    def interface(self, request):
        request.setHeader('Access-Control-Allow-Origin', '*')
        request.setHeader('Access-Control-Allow-Methods', 'GET')
        request.setHeader('Access-Control-Allow-Headers', '*')
        request.setHeader('Access-Control-Max-Age', 2520)  # 42 hours

        return json.dumps(self.input_interface)

    @klein_app.route('/predict', methods=['POST'])
    @inlineCallbacks
    def predict(self, request):
        request.setHeader('Access-Control-Allow-Origin', '*')
        request.setHeader('Access-Control-Allow-Methods', 'POST')
        request.setHeader('Access-Control-Allow-Headers', '*')
        request.setHeader('Access-Control-Max-Age', 2520)  # 42 hours

        logger.info('predicting...')

        result = yield threads.deferToThread(self.received_data_handler, request)

        returnValue(result)

    @klein_app.route('/responses', methods=['POST'])
    @inlineCallbacks
    def responses(self, request):
        request.setHeader('Access-Control-Allow-Origin', '*')
        request.setHeader('Access-Control-Allow-Methods', 'POST')
        request.setHeader('Access-Control-Allow-Headers', '*')
        request.setHeader('Access-Control-Max-Age', 2520)  # 42 hours

        logger.info('getting responses...')

        result = yield threads.deferToThread(self.responses_data_handler, request)

        returnValue(result)

    def responses_data_handler(self, request):
        req_id = request.args['request_id'][0]

        print(req_id)
        print(self.reqest_list)

        if req_id not in self.reqest_list:
            response = [{
                'type': 'PlainText',
                'content': 'The results of request {} cannot be retrieved'.format(req_id),
                'label': ''
            }]

            return json.dumps(response)

        try:
            response = self.result_dict[req_id]
            #removing content of list and dict
            del self.result_dict[req_id]
            self.reqest_list.remove(req_id)
        except KeyError:
            response = [{'type': 'DelayedResponse', 'request_id': req_id}]


        return json.dumps(response)
