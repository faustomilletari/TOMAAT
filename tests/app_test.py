import logging
from tomaat.server import TomaatApp


def pre_processing_mock_function(data):
    data['field'] = 0

    return data


def post_processing_mock_function(data):
    data['field'] += 1

    return data


def inference_mock_function(data):
    data['field'] += 10

    return data


mock_app = TomaatApp(
    preprocess_fun=pre_processing_mock_function,
    inference_fun=inference_mock_function,
    postprocess_fun=post_processing_mock_function
)


def test_tomaatapp_answer():
    logging.info('TESTING TOMAAT APP INIT FUNCTION')
    assert mock_app.preprocess_fun == pre_processing_mock_function
    assert mock_app.inference_fun == inference_mock_function
    assert mock_app.postprocess_fun == post_processing_mock_function


def test_tomaatapp_basic_functionality():
    data = {}

    result = mock_app(data)

    return result


def test_tomaatapp_basic_functionality_answer():
    logging.info('TESTING TOMAAT APP BASIC FUNCTIONALITY')

    result = test_tomaatapp_basic_functionality()

    assert isinstance(result, dict)
    assert len(result.keys()) == 1
    assert list(result.keys())[0] == 'field'
    assert result['field'] == 11


