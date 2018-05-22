import click
import json

from tomaat.server import TomaatApp, TomaatService
from tomaat.frameworks.tf import Prediction
from tomaat.extras import TransformChain
from tomaat.extras import (
    FromSITKToNumpy,
    FromITKFormatFilenameToSITK,
    FromNumpyOriginalSizeToStandardSize,
    FromSITKOriginalIntensitiesToRescaledIntensities,
    FromListToNumpy5DArray,
    FromSITKUint8ToSITKFloat32,
    FromSITKOriginalResolutionToStandardResolution,
    ThresholdNumpy,
    FromNumpyToSITK,
    FromNumpyStandardSizeToOriginalSize,
    FromNumpy5DArrayToList,
    FromSITKStandardResolutionToOriginalResolution,
)


input_interface = \
    [
        {'type': 'volume', 'destination': 'images'},
        {'type': 'slider', 'destination': 'threshold', 'minimum': 0, 'maximum': 1},
        {'type': 'checkbox', 'destination': 'RAS', 'text': 'use slicer coordinate conventions'},
        {'type': 'radiobutton', 'destination': 'spacing_metric', 'text': 'spacing metric', 'options': ['millimeters', 'meters']},
    ]

output_interface = \
    [
        {'type': 'LabelVolume', 'field': 'images'}
    ]


def create_pre_process_pipeline(config):
    """
    This is an application-specific function that creates the pre_processing pipeline. it returns a callable.
    :type config: dict a dictionary containing the configuration for this specific task
    :return: a callable
    """
    transform_1 = FromITKFormatFilenameToSITK(fields=['images'])
    transform_2 = FromSITKUint8ToSITKFloat32(fields=['images'])
    transform_3 = FromSITKOriginalIntensitiesToRescaledIntensities(fields=['images'])
    transform_4 = FromSITKOriginalResolutionToStandardResolution(
        fields=['images'],
        resolution=config['volume_resolution'],
        field_spacing_metric='spacing_metric'
    )
    transform_5 = FromSITKToNumpy(fields=['images'])
    transform_6 = FromNumpyOriginalSizeToStandardSize(fields=['images'], size=config['volume_size'])
    transform_7 = FromListToNumpy5DArray(fields=['images'])

    pre_process_pipeline = TransformChain(
        [transform_1,
         transform_2,
         transform_3,
         transform_4,
         transform_5,
         transform_6,
         transform_7,
         ]
    )

    return pre_process_pipeline


def create_post_process_pipeline(config):
    """
    This is an application-specific function that creates the post_processing pipeline. it returns a callable.
    :type config: dict a dictionary containing the configuration for this specific task
    :return: a callable
    """
    antitransform_8 = ThresholdNumpy(image_field='images', threshold_field='threshold')
    antitransform_7 = FromNumpy5DArrayToList(fields=['images'])
    antitransform_6 = FromNumpyStandardSizeToOriginalSize(fields=['images'])
    antitransform_5 = FromNumpyToSITK(fields=['images'])
    antitransform_4 = FromSITKStandardResolutionToOriginalResolution(fields=['images'])

    post_process_pipeline = TransformChain(
        [antitransform_8,
         antitransform_7,
         antitransform_6,
         antitransform_5,
         antitransform_4,
         ]
    )

    return post_process_pipeline


@click.group()
def cli():
    pass


@click.command()
@click.option('--model_path')
@click.option('--config_file_path')
def start_service(model_path, config_file_path):
    """

    :type model_path: str valid path of the TF model where a .pb model file is present
    :type config_file_path: str valid path of the config file in JSON format
    :return: None
    """
    with open(config_file_path) as f:
        config = json.load(f)

    pre_process_pipeline = create_pre_process_pipeline(config)
    post_process_pipeline = create_post_process_pipeline(config)

    tensorflow_prediction = Prediction(
        model_path=model_path,
        input_tensors_names=["images:0"],
        input_fields=["images"],
        output_tensors_names=["logits:0"],
        output_fields=["images"]  # replace images with results because it's more convenient for the transforms
    )

    application = TomaatApp(
        preprocess_fun=pre_process_pipeline,
        inference_fun=tensorflow_prediction,
        postprocess_fun=post_process_pipeline
    )

    service = TomaatService(
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
