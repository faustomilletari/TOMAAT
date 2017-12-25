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
from werkzeug import secure_filename

import tempfile


ANNOUNCEMENT_SERVER_URL = 'http://tomaat.cloud:8000/announce'
ANNOUNCEMENT_INTERVAL = 1600  # seconds
INFERENCE_INTERVAL = 0.1

OWN_PORT = 9000


def do_announcement(announcement_server_url, message):
    json_message = json.dumps(message)

    response = requests.post(announcement_server_url, data=json_message)

    response_json = response.json()

    if response_json['status'] != 0:
        print 'status {}'.format(response_json['status'])
        print 'errors: {}'.format(response_json['error'])
        raise SystemError


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

        self.app.run(port=self.port, host='0.0.0.0')

    @app.route('/', methods=['POST'])
    @inlineCallbacks
    def predict(self, request):
        print 'predicting...'

        result = yield threads.deferToThread(self.received_data_handler, request)

        returnValue(result)

    def add_inference_looping_call(self, fun, args, delay=INFERENCE_INTERVAL):
        self.inference_looping_call.append(LoopingCall(fun, *args))
        index = len(self.inference_looping_call) - 1
        self.inference_looping_call[index].start(delay)

        return index

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
            return -1

        try:
            host = self.params['host']
        except KeyError:
            ip = urlopen('http://ip.42.pl/raw').read()
            port = 9000
            host = 'http://' + ip + ':' + str(port) + '/'
            pass

        message = {
            'api_key': api_key,
            'host': host,
            'modality': self.params['modality'],
            'anatomy': self.params['anatomy'],
            'description': self.params['description'],
        }

        self.announcement_looping_call.append(LoopingCall(fun, (announcement_server_url, message)))
        index = len(self.announcement_looping_call) - 1
        self.announcement_looping_call[index].start(delay)

        return index

    def stop_announcement_looping_call(self, index):
        self.announcement_looping_call[index].stop()

    def do_inference(self, input_data, **kwargs):
        raise NotImplementedError

    def received_data_handler(self, request):
        status = 0
        error = ''
        json_data = json.loads(request.args['json'][0])

        savepath = tempfile.gettempdir()

        print 'RECEIVED REQUEST'

        uid = uuid.uuid4()

        mha_file = str(uid) + '.mha'
        mha_seg = str(uid) + '_seg.mha'

        tmp_filename_mha = os.path.join(savepath, mha_file)
        tmp_segmentation_mha = os.path.join(savepath, mha_seg)

        with open(tmp_filename_mha, 'wb') as f:
            f.write(request.args[b'file'][0])

        data = {self.image_field: [tmp_filename_mha], 'uids': [uid]}

        transformed_data = self.data_read_pipeline(data)

        result = self.do_inference(transformed_data, json_data['threshold'])

        try:
            elapsed_time = result['elapsed_time'] * 1000  # now it is milliseconds
        except KeyError:
            elapsed_time = -1

        print 'INFERENCE DONE'

        transformed_result = self.data_write_pipeline(result)

        filename = os.path.join(savepath, tmp_segmentation_mha)
        writer = sitk.ImageFileWriter()
        writer.SetFileName(filename)
        writer.SetUseCompression(True)
        writer.Execute(transformed_result[self.segmentation_field][0])

        print 'WRITING BACK'

        with open(filename, 'rb') as f:
            message = json.dumps({
                'content_mha': base64.encodestring(f.read()),
                'error': error,
                'status': status,
                'time': elapsed_time,

            })

        print 'SENDING INFERENCE RESULTS BACK'

        os.remove(tmp_filename_mha)
        os.remove(tmp_segmentation_mha)

        return message


