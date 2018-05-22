class TransformChain(object):
    def __init__(self, transforms_list):
        super(TransformChain, self).__init__()
        self.transforms_list = transforms_list

    def __call__(self, data):
        for transform in self.transforms_list:
            data = transform(data)

        return data
