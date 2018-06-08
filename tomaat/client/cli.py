import qprompt


def get_server_interface():


def direct_connection():
    hostname = qprompt.ask_str("Enter server hostname (Eg. http://localhost:9000/)")

    if hostname[-1] != '/':
        hostname += '/'

    interface = get_server_interface(hostname + 'interface')


menu = qprompt.Menu()
menu.add("d", "Direct connection to server", direct_connection)
menu.add("l", "List publicly available services ", obtain_public_list)
choice = menu.show()