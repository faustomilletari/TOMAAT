from OpenSSL import crypto, SSL
from socket import gethostname
from pprint import pprint
from time import gmtime, mktime
from os.path import exists

def create_self_signed_cert(CERT_FILE = "./tomaat.crt",KEY_FILE = "./tomaat.key"):
    if not exists(CERT_FILE) or not exists(KEY_FILE):
        # create a key pair
        k = crypto.PKey()
        k.generate_key(crypto.TYPE_RSA, 4096)

        # create a self-signed cert
        cert = crypto.X509()
        cert.get_subject().C = "US"
        cert.get_subject().ST = "TOMAAT"
        cert.get_subject().L = "TOMAAT"
        cert.get_subject().O = "TOMAAT"
        cert.get_subject().OU = "TOMAAT"
        cert.get_subject().CN = "*"
        cert.set_serial_number(1000)
        cert.gmtime_adj_notBefore(0)
        cert.gmtime_adj_notAfter(10*365*24*60*60)
        cert.set_issuer(cert.get_subject())
        cert.set_pubkey(k)
        cert.sign(k, 'sha256')
        cert.sign(k, 'sha1')

        # write keys
        with open(CERT_FILE, "wb") as f:
            f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
        with open(KEY_FILE, "wb") as f:
            f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, k))

def get_cert_fingerprint(CERT_FILE):
    with open(CERT_FILE,"rb") as c:
        x509 = crypto.load_certificate(crypto.FILETYPE_PEM,c.read())
    cert_hash = x509.digest("sha256").decode("ASCII").upper()
    return cert_hash
