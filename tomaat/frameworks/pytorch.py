import torch
import torch.backends.cudnn as cudnn
import numpy as np


class Prediction(object):
    def __init__(self, model_path, input_arg_names, input_fields, output_fields, with_gpu=True):
        super(Prediction, self).__init__()

        self.model = torch.load(model_path)
        self.with_gpu = with_gpu

        if self.with_gpu:
            self.model.cuda()
            # avoid nonsense from cudnn
            cudnn.enabled = True
            cudnn.benchmark = True

        self.input_fields = input_fields
        self.input_arg_names = input_arg_names

        assert len(self.input_fields) == len(self.input_arg_names)

        self.output_fields = output_fields

    def __call__(self, data):
        arg_dict = {}

        for arg_name, field_name in zip(self.input_arg_names, self.input_fields):
            arg_dict[arg_name] = torch.from_numpy(data[field_name])

            if self.with_gpu:
                arg_dict[arg_name].cuda()

        outputs = self.model(**arg_dict)

        if not hasattr(outputs, 'len'):
            outputs = [outputs]

        assert len(outputs) == len(self.output_fields)

        for output, output_field in zip(outputs, self.output_fields):
            data[output_field] = output.cpu().detach().numpy().astype(dtype=np.float32)

        return data