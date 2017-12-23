import click
import os

from tinydb import TinyDB, Query
from binascii import hexlify


key_length = 64


@click.group()
def cli():
    pass


@click.command()
@click.option('--api_key')
@click.option('--db_filename', default='./db/api_key_db.json')
@click.option('--remove', default=False)
def add_db_record(api_key, db_filename, remove):
    db = TinyDB(db_filename)
    if len(api_key) == key_length:
        if not remove:
            db.insert({'api_key': api_key})
        else:
            API_KEY = Query()
            el = db.get(API_KEY.api_key == api_key)
            if el is not None:
                db.remove(doc_ids=[el.doc_id])
    db.close()


@click.command()
def create_api_key():
    print hexlify(os.urandom(key_length/2))


cli.add_command(add_db_record)
cli.add_command(create_api_key)


if __name__ == '__main__':
    cli()