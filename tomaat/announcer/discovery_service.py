import json
from klein import Klein
from twisted.internet.defer import inlineCallbacks, returnValue
from twisted.internet import reactor
from twisted.internet import threads
import time
import click

from tinydb import TinyDB, Query


timeout = 3600  # seconds
port = 80

db_service_endpoints = []

app = Klein()


@click.group()
def cli():
    pass


@click.command()
@click.option('--db_filename', default='./db/api_key_db.json')
def start_service(db_filename):
    global db_api_keys
    db_api_keys = TinyDB(db_filename)

    app.run(port=port, host='0.0.0.0')
    reactor.run()


def screen_announcement_json(json_data):
    status = 0
    error = ''

    # message is in the format:
    # 'api_key' -- is the API key assigned to the host of the DL service
    # 'host' -- is the hostname of the endpoint where the service runs
    # 'port' -- is the port of the endpoint where the service runs
    # 'modality' -- is the modality
    # 'anatomy' -- is the anatomy
    # 'description' -- is a short textual description of the service

    try:
        json_data['api_key']
    except KeyError:
        status += 1
        error += 'No api_key specified '

    try:
        json_data['host']
    except KeyError:
        status += 1
        error += 'No endpoint hostname specified '

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
        json_data['description']
    except KeyError:
        status += 1
        error += 'No endpoint description specified '

    return status, error


def announce_handler(json_data):
    status, error = screen_announcement_json(json_data)
    if status == 0:
        print 'CHECKING API KEY'
        service_api_key = json_data['api_key']

        API_KEY = Query()
        api_key_list = db_api_keys.search(API_KEY.api_key == service_api_key)

        if api_key_list is None:
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
    endpoint_hosts = []
    endpoint_modalities = []
    endpoint_anatomies = []
    endpoint_descriptions = []

    print 'SENDING CURRENT ENDPOINTS'
    print 'ENDPOINTS IN MEMORY ARE: {}'.format(db_service_endpoints)

    for element in db_service_endpoints:
        print element
        current_time = time.time()
        if (current_time - element['creation_time']) < timeout:
            endpoint_hosts.append(str(element['host']))
            endpoint_modalities.append(str(element['modality']))
            endpoint_anatomies.append(str(element['anatomy']))
            endpoint_descriptions.append(str(element['description']))

    print "RETURNING {} SERVICES".format(len(endpoint_hosts))

    print endpoint_hosts

    message = {
        'hosts': endpoint_hosts,
        'modalities': endpoint_modalities,
        'anatomies': endpoint_anatomies,
        'descriptions': endpoint_descriptions
    }

    return json.dumps(message)


@app.route('/announce', methods=['POST'])
@inlineCallbacks
def announce(request):
    data = json.loads(request.content.read())
    result = yield threads.deferToThread(announce_handler, data)

    returnValue(result)


@app.route('/discover')
@inlineCallbacks
def discover(none):
    result = yield threads.deferToThread(discover_handler)

    returnValue(result)


if __name__ == '__main__':
    cli()


