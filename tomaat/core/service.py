import json
import os
import uuid
import requests
import base64
import SimpleITK as sitk

from urllib2 import urlopen
from klein import Klein
from twisted.internet.defer import inlineCallbacks, returnValue, DeferredLock
from twisted.internet import threads
from twisted.internet.task import LoopingCall
from twisted.internet import reactor
from twisted.logger import Logger

import tempfile


ANNOUNCEMENT_SERVER_URL = 'http://tomaat.cloud:8000/announce'
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


class TOMAATService(object):
    app = Klein()
    announcement_looping_call = []

    gpu_lock = DeferredLock()

    def __init__(self,
                 params,
                 data_read_pipeline,
                 data_write_pipeline,
                 image_field='images',
                 segmentation_field='label',
                 port=OWN_PORT
                 ):
        super(TOMAATService, self).__init__()

        self.params = params
        self.data_read_pipeline = data_read_pipeline
        self.data_write_pipeline = data_write_pipeline

        self.image_field = image_field
        self.segmentation_field = segmentation_field
        self.port = port

    @app.route('/interface', methods=['GET'])
    def interface(self, _):
        widgets = self.widgets
        return json.dumps(widgets)

    @app.route('/predict', methods=['POST'])
    @inlineCallbacks
    def predict(self, request):
        print 'predicting...'

        result = yield threads.deferToThread(self.received_data_handler, request)

        returnValue(result)

    def add_announcement_looping_call(
            self,
            fun=do_announcement,
            announcement_server_url=ANNOUNCEMENT_SERVER_URL,
            delay=ANNOUNCEMENT_INTERVAL
    ):
        try:
            api_key = self.params['api_key']
        except KeyError:
            raise ValueError('Api-key is missing')

        try:
            host = self.params['host']
        except KeyError:
            ip = urlopen('http://ip.42.pl/raw').read()
            port = 9000
            host = 'http://' + str(ip) + ':' + str(port) + '/'
            pass

        message = {
            'api_key': api_key,
            'prediction_url': host+'predict',
            'interface_url': host+'interface',
            'name': self.params['name'],
            'modality': self.params['modality'],
            'task': self.params['task'],
            'anatomy': self.params['anatomy'],
            'description': self.params['description'],
        }

        self.announcement_looping_call = LoopingCall(fun, *(announcement_server_url, message))

        self.announcement_looping_call.start(delay)

    def stop_announcement_looping_call(self, index):
        self.announcement_looping_call[index].stop()

    def do_inference(self, input_data):
        raise NotImplementedError

    def parse_request(self, request):
        raise NotImplementedError

    def prepare_response(self, result):
        raise NotImplementedError

    def received_data_handler(self, request):
        print 'PARSING REQUEST'
        data = self.parse_request(request)

        print 'TRANSFORMING DATA'
        transformed_data = self.data_read_pipeline(data)

        print 'DOING INFERENCE'
        self.gpu_lock.acquire()  # acquire GPU lock
        result = self.do_inference(transformed_data)  # GPU call
        self.gpu_lock.release()  # release GPU lock

        print 'TRANSFORMING RESULTS'
        transformed_result = self.data_write_pipeline(result)

        print 'PREPARING RESPONSE'
        response = self.prepare_response(transformed_result)

        message = json.dumps(response)

        return message

    def run(self):
        self.app.run(port=self.port, host='0.0.0.0')
        reactor.run()


