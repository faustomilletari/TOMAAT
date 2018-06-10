import click
import json
import os
import nibabel as nib
import SimpleITK as sitk
import numpy as np

from tomaat.server import TomaatServiceDelayedResponse


# this extra works with model zoo from NiftyNet and in particular with highresnet3d_brain_parcellation model.


input_interface = \
    [
        {'type': 'volume', 'destination': 'T1'},
        {'type': 'checkbox', 'destination': 'alt_convention', 'text': 'use alternative coordinate convention'},
    ]

output_interface = \
    [
        {'type': 'LabelVolume', 'field': 'labels'}
    ]


class FromNumpyToSITK(object):
    def __init__(self,
                 fields,
                 field_original_spacing='original_spacings_NP',
                 field_original_direction='original_directions_NP',
                 field_original_origins='original_origins_NP'
                 ):
        '''
        FromNumpyToSITK converts numpy array data in SITK data
        :param fields: fields of the dictionary whose content should be modified
        :param field_original_spacing: field data dictionary used to store original spacing of sitk volumes
        :param field_original_direction: field data dictionary used to store original direction of sitk volumes
        :param field_original_origins: field data dictionary used to store original origin coordinate of sitk volumes
        '''
        super(FromNumpyToSITK, self).__init__()
        self.fields = fields
        self.field_original_spacing = field_original_spacing
        self.field_original_direction = field_original_direction
        self.field_original_origins = field_original_origins

    def __call__(self, data):
        original_spacings = data[self.field_original_spacing]
        original_directions = data[self.field_original_direction]
        original_origins = data[self.field_original_origins]
        for field in self.fields:
            for i in range(len(data[field])):
                numpy_data = np.transpose(data[field][i], [2, 1, 0])

                print(original_spacings[field][i].tolist())

                data[field][i] = sitk.GetImageFromArray(numpy_data)
                data[field][i].SetDirection(original_directions[field][i])
                data[field][i].SetOrigin(original_origins[field][i])
                data[field][i].SetSpacing(original_spacings[field][i].tolist())

        return data


class FromNiftyToNumpy(object):
    def __init__(self, field):
        self.field = field

    def __call__(self, data):
        entries = []
        spacings = []
        origins = []
        directions = []

        for entry, convention in zip(data[self.field], data['alt_convention']):
            nii_image = nib.load(entry)

            spacing = np.asarray([1., 1., 1.])
            affine = nii_image.header.get_sform()

            spacings.append(spacing)
            if convention == 'True':
                directions.append((affine[0:3, 0:3]).flatten().tolist())
                origins.append(affine[0:3, 3].flatten().tolist())
            else:
                affine = np.matmul(np.diag([-1, -1, 1, 1]), affine)
                directions.append(affine[0:3, 0:3].flatten().tolist())
                origins.append(affine[0:3, 3].flatten().tolist())

            numpy_image = np.squeeze(nii_image.get_data().astype(np.float32))

            entries.append(numpy_image)

        data[self.field] = entries
        data['spacings'] = {self.field: spacings}
        data['origins'] = {self.field: origins}
        data['directions'] = {self.field: directions}


        return data


class NiftyNetZooApp(object):
    def __init__(self, ini_file):
        self.ini_file = ini_file

    def __call__(self, data, gpu_lock):
        # lock mechanism

        with open(self.ini_file, 'r') as f:
            txt = f.read()

        dirname = os.path.dirname(data['T1'][0])

        txt = txt.replace('<dir>', dirname)

        new_conf = os.path.join(dirname, './default.ini')

        with open(new_conf, 'w+') as f:
            f.write(txt)

        command = 'net_segment inference -c {}'.format(new_conf)

        if gpu_lock is not None:
            gpu_lock.acquire()  # acquire GPU lock

        os.system(command)

        if gpu_lock is not None:
            gpu_lock.release()  # release GPU lock

        transform_1 = FromNiftyToNumpy(field='labels')
        transform_2 = FromNumpyToSITK(
            fields=['labels'],
            field_original_direction='directions',
            field_original_spacing='spacings',
            field_original_origins='origins',
        )

        img_basename = os.path.splitext(data['T1'][0])[0].replace("-", "")

        label_name = img_basename + '_niftynet_out.nii.gz'

        data['labels'] = [img_basename + '_niftynet_out.nii.gz']

        data = transform_2(transform_1(data))

        os.remove(new_conf)
        os.remove(data['T1'][0])
        os.remove(label_name)

        return data


@click.group()
def cli():
    pass


@click.command()
@click.option('--config_file_path')
@click.option('--ini_file_path')
def start_service(config_file_path, ini_file_path):
    """
    :type config_file_path: str valid path of the config file in JSON format
    :type ini_file_path: str valid path of the config ini file for the niftynet model
    :return: None
    """
    with open(config_file_path) as f:
        config = json.load(f)

    application = NiftyNetZooApp(ini_file_path)

    service = TomaatServiceDelayedResponse(
        config=config,
        app=application,
        input_interface=input_interface,
        output_interface=output_interface
    )

    if config['announce']:
        service.start_service_announcement()

    service.run()


cli.add_command(start_service)


if __name__ == '__main__':
    cli()
