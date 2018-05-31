import json
import os
import copy
import requests
from klein import Klein
from twisted.internet.defer import inlineCallbacks, returnValue
from twisted.internet import reactor
from twisted.internet import threads
import time
import click

from tinydb import TinyDB, Query


timeout = 3600  # seconds

db_service_endpoints = []

app = Klein()

ssl_app = Klein()


@click.group()
def cli():
    pass


def screen_announcement_json(json_data):
    status = 0
    error = ''

    # message is in the format:
    # 'api_key' -- is the API key assigned to the host of the DL service

    # 'prediction_url' -- is the URL of the predictions service
    # 'interface_url' -- is the URL of the interface specification service
    # 'name' -- service name

    # 'modality' -- is the modality
    # 'anatomy' -- is the anatomy
    # 'task' -- is the task
    # 'description' -- is a short textual description of the service

    try:
        json_data['api_key']
    except KeyError:
        status += 1
        error += 'No api_key specified '

    try:
        json_data['prediction_url']
    except KeyError:
        status += 1
        error += 'No endpoint prediction url specified '

    try:
        json_data['interface_url']
    except KeyError:
        status += 1
        error += 'No endpoint interface url specified '

    try:
        json_data['name']
    except KeyError:
        status += 1
        error += 'No endpoint name specified '

    try:
        json_data['modality']
    except KeyError:
        status += 1
        error += 'No endpoint modality specified '

    try:
        json_data['anatomy']
    except KeyError:
        status += 1
        error += 'No endpoint anatomy specified '

    try:
        json_data['task']
    except KeyError:
        status += 1
        error += 'No endpoint task specified '

    try:
        json_data['description']
    except KeyError:
        status += 1
        error += 'No endpoint description specified '

    return status, error


def open_db_api_keys(file):
    global db_api_keys
    db_api_keys = TinyDB(file)


def announce_handler(json_data):
    status, error = screen_announcement_json(json_data)

    if status == 0:
        print 'CHECKING API KEY'
        service_api_key = json_data['api_key']

        API_KEY = Query()
        api_key_list = db_api_keys.search(API_KEY.api_key == service_api_key)

        if (api_key_list is None) or (len(api_key_list) == 0):
            status += 1
            error += 'Your API KEY has not been recognized. Contact the administrator of the announcement service '
            print error

    if status == 0:
        print 'CREATING SERVICE'
        creation_time = time.time()

        idx_delete = None

        for i in range(len(db_service_endpoints)):
            # if the developer already has a service replace old service with new
            if db_service_endpoints[i]['api_key'] == json_data['api_key']:
                idx_delete = i

        if idx_delete is not None:
            del db_service_endpoints[idx_delete]

        json_data['creation_time'] = creation_time

        db_service_endpoints.append(json_data)

    message = {
        'status': status,
        'error': error,
    }

    message = json.dumps(message)

    return message


def discover_handler():
    print 'SENDING CURRENT ENDPOINTS'

    endpoint_list = []

    for element in db_service_endpoints:
        current_time = time.time()
        if (current_time - element['creation_time']) < timeout:
            element = copy.deepcopy(element)
            element['SID'] = element['api_key'][0:7]
            element['api_key'] = ''

            endpoint_list.append(element)

    return json.dumps(endpoint_list)


def discover_ssl_handler():
    print 'MAKING UNSECURE REQUEST TO LOCALHOST'

    response = requests.get('http://127.0.0.1:{}/discover'.format(unsecure_port))
    json_data = response.json()

    return json.dumps(json_data)


def announce_ssl_handler(json_data):
    print 'MAKING UNSECURE REQUEST TO LOCALHOST'

    response = requests.post('http://127.0.0.1:{}/announce'.format(unsecure_port), data=json.dumps(json_data))
    json_data = response.json()

    return json.dumps(json_data)


@app.route('/announce', methods=['POST'])
@inlineCallbacks
def announce(request):
    data = json.loads(request.content.read())
    result = yield threads.deferToThread(announce_handler, data)

    returnValue(result)


@app.route('/discover')
@inlineCallbacks
def discover(request):
    request.setHeader('Access-Control-Allow-Origin', '*')
    request.setHeader('Access-Control-Allow-Methods', 'GET')
    request.setHeader('Access-Control-Allow-Headers', 'x-prototype-version,x-requested-with')
    request.setHeader('Access-Control-Max-Age', 2520)  # 42 hours

    result = yield threads.deferToThread(discover_handler)

    returnValue(result)


@ssl_app.route('/discover')
@inlineCallbacks
def discover_ssl(request):
    request.setHeader('Access-Control-Allow-Origin', '*')
    request.setHeader('Access-Control-Allow-Methods', 'GET')
    request.setHeader('Access-Control-Allow-Headers', 'x-prototype-version,x-requested-with')
    request.setHeader('Access-Control-Max-Age', 2520)  # 42 hours

    result = yield threads.deferToThread(discover_ssl_handler)

    returnValue(result)


@ssl_app.route('/announce', methods=['POST'])
@inlineCallbacks
def announce_ssl(request):
    data = json.loads(request.content.read())
    result = yield threads.deferToThread(announce_ssl_handler, data)

    returnValue(result)


@click.command()
@click.option('--db_filename', default='./db/api_key_db.json')
@click.option('--port', default=8001)
def start_service(db_filename, port):
    open_db_api_keys(db_filename)
    app.run(port=port, host='0.0.0.0')

    reactor.run()


@click.command()
@click.option('--port', default=7001)
@click.option('--ssl_cert_dir', default='./ssl')
@click.option('--http_port', default=8001)
def start_ssl_service(port, ssl_cert_dir, http_port):
    key = os.path.join(ssl_cert_dir,"key.pem")
    cert = os.path.join(ssl_cert_dir,"cert.pem")
    spec_template = "ssl:{}:privateKey={}:certKey={}"
    spec = spec_template.format(port, key, cert)

    global unsecure_port
    unsecure_port = http_port

    ssl_app.run(endpoint_description=spec, host='0.0.0.0', port=port)

    reactor.run()


cli.add_command(start_service)
cli.add_command(start_ssl_service)


if __name__ == '__main__':
    cli()


