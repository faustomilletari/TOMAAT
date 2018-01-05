import tensorflow as tf
import numpy as np
import click
import time
from ..core.service import TOMAATService
from ..core.utils import TransformChain
from ..core.transforms import (
    FromSITKToNumpy,
    FromITKFormatFilenameToSITK,
    FromNumpyOriginalSizeToStandardSize,
    FromSITKOriginalIntensitiesToRescaledIntensities,
    FromListToNumpy5DArray,
    FromSITKUint8ToSITKFloat32,
    FromSITKOriginalResolutionToStandardResolution,

    FromNumpyToSITK,
    FromNumpyStandardSizeToOriginalSize,
    FromNumpy5DArrayToList,
    FromSITKStandardResolutionToOriginalResolution,
)


@click.group()
def cli():
    pass


class TOMAATTensorflow(TOMAATService):
    def __init__(self, sess, input_tensor, output_tensor, **kwargs):

        self.sess = sess
        self.input_tensor = input_tensor
        self.output_tensor = output_tensor

        super(TOMAATTensorflow, self).__init__(**kwargs)

    def do_inference(self, data, threshold=0.5):
        start_time = time.time()
        result = self.sess.run(fetches=self.output_tensor, feed_dict={self.input_tensor: data[self.image_field]})
        elap_time = time.time() - start_time

        data[self.segmentation_field] = (result > threshold).astype(np.float32)
        data['elapsed_time'] = elap_time

        return data


def inference_function(sess,
                       input_tensor,
                       output_tensor,
                       data_queue,
                       result_queue,
                       image_field,
                       segmentation_field,
                       ):
    try:
        data = data_queue.pop()
    except IndexError:
        return

    print 'DOING INFERENCE'

    result = sess.run(fetches=output_tensor, feed_dict={input_tensor: data[image_field]})

    result_dict = {}
    result_dict['uid'] = data['uid']
    result_dict[segmentation_field] = [result]

    result_queue.append_left(result_dict)

    return


@click.command()
@click.option('--model_path')
@click.option('--input_tensor_name', default="images:0")
@click.option('--output_tensor_name', default="logits:0")
@click.option('--port', default=9000)
@click.option('--announce', default=False)
@click.option('--api_key', default='')
@click.option('--modality', default='None')
@click.option('--anatomy', default='None')
@click.option('--description', default='None')
@click.option('--volume_resolution')
@click.option('--volume_size')
def start_prediction_service(
        model_path,
        input_tensor_name,
        output_tensor_name,
        port,
        announce,
        api_key,
        modality,
        anatomy,
        description,
        volume_resolution,
        volume_size
):
    volume_size = eval(volume_size)
    volume_resolution = eval(volume_resolution)

    params = {
        'port': port,
        'api_key': api_key,
        'modality': modality,
        'anatomy': anatomy,
        'description': description,
        'volume_resolution': volume_resolution,
        'volume_size': volume_size,
    }
    sess = tf.Session()

    _ = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], model_path)

    graph = tf.get_default_graph()

    input_tensor = graph.get_tensor_by_name(input_tensor_name)
    output_tensor = graph.get_tensor_by_name(output_tensor_name)

    transform_1 = FromITKFormatFilenameToSITK(fields=['images'])
    transform_2 = FromSITKUint8ToSITKFloat32(fields=['images'])
    transform_3 = FromSITKOriginalIntensitiesToRescaledIntensities(fields=['images'])
    transform_4 = FromSITKOriginalResolutionToStandardResolution(fields=['images'], resolution=volume_resolution)
    transform_5 = FromSITKToNumpy(fields=['images'])
    transform_6 = FromNumpyOriginalSizeToStandardSize(fields=['images'], size=volume_size)
    transform_7 = FromListToNumpy5DArray(fields=['images'])

    antitransform_7 = FromNumpy5DArrayToList(fields=['images'])
    antitransform_6 = FromNumpyStandardSizeToOriginalSize(fields=['images'])
    antitransform_5 = FromNumpyToSITK(fields=['images'])
    antitransform_4 = FromSITKStandardResolutionToOriginalResolution(fields=['images'])

    data_read_pipeline = TransformChain(
        [transform_1,
         transform_2,
         transform_3,
         transform_4,
         transform_5,
         transform_6,
         transform_7,
         ]
    )

    data_write_pipeline = TransformChain(
        [antitransform_7,
         antitransform_6,
         antitransform_5,
         antitransform_4,
         ]
    )

    service = TOMAATTensorflow(
        sess=sess,
        input_tensor=input_tensor,
        output_tensor=output_tensor,
        params=params,
        data_read_pipeline=data_read_pipeline,
        data_write_pipeline=data_write_pipeline,
        image_field='images',
        segmentation_field='images',
        port=port
    )

    if announce:
        service.add_announcement_looping_call()

    service.run()


cli.add_command(start_prediction_service)


if __name__ == '__main__':
    cli()