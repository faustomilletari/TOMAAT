import tensorflow as tf


class Prediction(object):
    def __init__(self, model_path, input_tensors_names, input_fields, output_tensors_names, output_fields):
        super(Prediction, self).__init__()

        self.sess = tf.Session()

        _ = tf.saved_model.loader.load(self.sess, [tf.saved_model.tag_constants.SERVING], model_path)

        self.graph = tf.get_default_graph()

        self.input_tensors = []

        for input_name in input_tensors_names:
            self.input_tensors.append(self.graph.get_tensor_by_name(input_name))

        self.input_fields = input_fields

        self.output_tensors = []

        for output_name in output_tensors_names:
            self.output_tensors.append(self.graph.get_tensor_by_name(output_name))

        self.output_fields = output_fields

        assert len(self.input_tensors) == len(self.input_fields)
        assert len(self.output_tensors) == len(self.output_fields)

    def __call__(self, data):
        feed_dict = {}

        for input_tensor, input_field in zip(self.input_tensors, self.input_fields):
            feed_dict[input_tensor] = data[input_field]

        outputs = self.sess.run(self.output_tensors, feed_dict=feed_dict)

        for output, output_field in zip(outputs, self.output_fields):
            data[output_field] = output

        return data
