import json
import os
import uuid
import requests
import base64
import SimpleITK as sitk

from cgi import FieldStorage
from urllib2 import urlopen
from klein import Klein
from twisted.internet.defer import inlineCallbacks, returnValue
from twisted.internet import threads
from twisted.internet.task import LoopingCall
from twisted.internet import reactor
from twisted.logger import Logger

import tempfile


ANNOUNCEMENT_SERVER_URL = 'http://10.110.21.48:8888/announce'
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

    def __init__(self,
                 params,
                 data_read_pipeline,
                 data_write_pipeline,
                 image_field='images',
                 segmentation_field='label',
                 port=OWN_PORT
                 ):
        super(TOMAATService, self).__init__()
        self.inference_looping_call = []
        self.announcement_looping_call = []
        self.params=params
        self.data_read_pipeline = data_read_pipeline
        self.data_write_pipeline = data_write_pipeline

        self.image_field = image_field
        self.segmentation_field = segmentation_field
        self.port = port

    @app.route('/interface', methods=['GET'])
    def interface(self, _):
        widgets = [
            {'type': 'volume', 'destination': 'input'},  # a volume that will be transmitted in field 'input'
            {'type': 'slider', 'destination': 'threshold', 'minimum': 0, 'maximum': 1}  # a threshold
        ]
        return json.dumps(widgets)

    @app.route('/predict', methods=['POST'])
    @inlineCallbacks
    def predict(self, request):
        print 'predicting...'

        result = yield threads.deferToThread(self.received_data_handler, request)

        returnValue(result)

    def stop_inference_looping_call(self, index):
        self.inference_looping_call[index].stop()

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
            host = 'http://' + '10.110.21.48' + ':' + str(port) + '/'
            pass

        message = {
            'api_key': api_key,
            'prediction_url': host+'predict',
            'interface_url': host+'interface',
            'name': self.params['name'],
            'modality': self.params['modality'],
            'dimensionality': self.params['dimensionality'],
            'anatomy': self.params['anatomy'],
            'description': self.params['description'],
            'SID': self.params['SID']
        }

        self.announcement_looping_call = LoopingCall(fun, *(announcement_server_url, message))

        self.announcement_looping_call.start(delay)

    def stop_announcement_looping_call(self, index):
        self.announcement_looping_call[index].stop()

    def do_inference(self, input_data, **kwargs):
        raise NotImplementedError

    def received_data_handler(self, request):
        savepath = tempfile.gettempdir()

        print 'RECEIVED REQUEST'

        uid = uuid.uuid4()

        mha_file = str(uid) + '.mha'
        mha_seg = str(uid) + '_seg.mha'

        tmp_filename_mha = os.path.join(savepath, mha_file)
        tmp_segmentation_mha = os.path.join(savepath, mha_seg)

        with open(tmp_filename_mha, 'wb') as f:
            f.write(request.args['input'][0])

        threshold = float(request.args['threshold'][0])

        data = {self.image_field: [tmp_filename_mha], 'uids': [uid]}

        transformed_data = self.data_read_pipeline(data)

        result = self.do_inference(transformed_data, threshold)

        print 'INFERENCE DONE'

        transformed_result = self.data_write_pipeline(result)

        filename = os.path.join(savepath, tmp_segmentation_mha)
        writer = sitk.ImageFileWriter()
        writer.SetFileName(filename)
        writer.SetUseCompression(True)
        writer.Execute(transformed_result[self.segmentation_field][0])

        print 'WRITING BACK'

        with open(filename, 'rb') as f:
           vol_string = base64.encodestring(f.read())

        data = {
            'descriptions': [
                {'type': 'LabelVolume'}
            ],
            'responses': [
                {'content': vol_string}
            ]
        }

        message = json.dumps(data)

        print 'SENDING INFERENCE RESULTS BACK'

        os.remove(tmp_filename_mha)
        os.remove(tmp_segmentation_mha)

        return message

    def run(self):
        self.app.run(port=self.port, host='0.0.0.0')
        reactor.run()


