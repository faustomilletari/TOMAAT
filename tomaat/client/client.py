import urlparse
import requests


class DirectConnectionManager(object):
    def __init__(self, hostname):
        self.hostname = hostname
        self.interface_url = urlparse.urljoin(hostname, 'interface')
        self.predict_url = urlparse.urljoin(hostname, 'predict')

    def get_prediction(self, message):
        pass

    def get_interface(self):
        response = requests.get(self.interface_url, timeout=5.0)
        interface = response.json()

        return interface


class PublicServerListManager(object):
    def __init__(self, list_url):
        self.list_url = list_url

    def get_server_list(self):
        response = requests.get(self.list_url, timeout=5.0)
        service_list = response.json()

        data = {}

        for service in service_list:
            data[service['modality']] = {}

        for service in service_list:
            data[service['modality']][service['anatomy']] = {}

        for service in service_list:
            data[service['modality']][service['anatomy']][service['task']] = []

        for service in service_list:
            data[service['modality']][service['anatomy']][service['task']].append(service)

        return data


class