import tempfile
import torch
import numpy as np
import uuid
import os

from tomaat.server import TomaatApp
from tomaat.frameworks.pytorch import Prediction


# test only pytorch support in frameworks.pytorch


modelpath = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))

os.makedirs(modelpath)

modelpath = os.path.join(modelpath, 'model.pt')


class MockNet(torch.nn.Module):
    def forward(self, input):
        output = torch.add(
            input,
            torch.from_numpy(np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32))
        )

        return output


net = MockNet()

torch.save(net, modelpath)

pred_object = Prediction(
    modelpath,
    input_arg_names=['input'],
    input_fields=['input_dict_field'],
    output_fields=['output_dict_field']
)


def test_pytorch_prediction():

    test_data = {'input_dict_field': np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], np.float32)}

    results_data = pred_object(test_data)

    return results_data


def test_pytorch_prediction_answer():
    results_data = test_pytorch_prediction()

    assert isinstance(results_data, dict)
    assert len(list(results_data.keys())) == 2  # now there is also the output dict field
    assert 'output_dict_field' in list(results_data.keys())

    assert isinstance(results_data['output_dict_field'], np.ndarray)

    assert np.all(results_data['output_dict_field'] == [2, 4, 6, 8, 10, 12, 14, 16, 18, 20])


# test app in this context


def pre_processing_mock_function(data):
    data['input_dict_field'] = \
        (np.asarray(data['input_dict_field']) + np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])).astype(np.float32)

    return data


def post_processing_mock_function(data):
    data['output_dict_field'] /= 2

    return data


mock_app = TomaatApp(
    preprocess_fun=pre_processing_mock_function,
    inference_fun=pred_object,
    postprocess_fun=post_processing_mock_function
)


def test_tomaatapp_pytorch_functionality():
    data = {'input_dict_field': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

    result = mock_app(data)

    return result


def test_tomaatapp_pytorch_functionality_answer():
    result = test_tomaatapp_pytorch_functionality()

    assert np.all(result['output_dict_field'] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

