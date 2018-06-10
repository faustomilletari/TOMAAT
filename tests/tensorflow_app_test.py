import tempfile
import tensorflow as tf
import numpy as np
import uuid
import os

from tomaat.server import TomaatApp
from tomaat.frameworks.tf import Prediction


# test only tensorflow support in frameworks.tf


modelpath = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))


def tf_build_graph():
    input = tf.placeholder(dtype=tf.float32, shape=(10), name='input')
    op = tf.add(input, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    output = tf.identity(op, name='output')

    return input, output


with tf.Session() as session:
    with tf.Graph().as_default():
        input, output = tf_build_graph()

        input_dict = {'input': tf.saved_model.utils.build_tensor_info(input)}
        output_dict = {'output': tf.saved_model.utils.build_tensor_info(output)}

        signature = tf.saved_model.signature_def_utils.build_signature_def(inputs=input_dict, outputs=output_dict)

        builder = tf.saved_model.builder.SavedModelBuilder(modelpath)

        builder.add_meta_graph_and_variables(session, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'prediction': signature,
            }
        )

        builder.save()


pred_object = Prediction(
    modelpath,
    input_tensors_names=['input:0'],
    input_fields=['input_dict_field'],
    output_tensors_names=['output:0'],
    output_fields=['output_dict_field']
)


def test_tensorflow_prediction():

    test_data = {'input_dict_field': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

    results_data = pred_object(test_data)

    return results_data


def test_tensorflow_prediction_answer():
    results_data = test_tensorflow_prediction()

    assert isinstance(results_data, dict)
    assert len(list(results_data.keys())) == 2  # now there is also the output dict field
    assert 'output_dict_field' in list(results_data.keys())

    assert isinstance(results_data['output_dict_field'], np.ndarray)

    assert np.all(results_data['output_dict_field'] == [2, 4, 6, 8, 10, 12, 14, 16, 18, 20])


# test app in this context


def pre_processing_mock_function(data):
    data['input_dict_field'] = np.asarray(data['input_dict_field']) + np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    return data


def post_processing_mock_function(data):
    data['output_dict_field'] /= 2

    return data


mock_app = TomaatApp(
    preprocess_fun=pre_processing_mock_function,
    inference_fun=pred_object,
    postprocess_fun=post_processing_mock_function
)


def test_tomaatapp_tensorflow_functionality():
    data = {'input_dict_field': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

    result = mock_app(data)

    return result


def test_tomaatapp_tensorflow_functionality_answer():
    result = test_tomaatapp_tensorflow_functionality()

    assert np.all(result['output_dict_field'] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

