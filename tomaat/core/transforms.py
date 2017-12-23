import SimpleITK as sitk
import numpy as np


class FromITKFormatFilenameToSITK(object):
    def __init__(self, fields):
        '''
        FromITKFormatFilenameToSITK replaces the content of the dictionary its methods are called onto stored in
        the fields 'fields' with SITK images.
        This transformation does not have a backward function (we don't want to overwrite our data, we rather create
        new data)
        :param fields: fields of the dictionary whose content should be replaced by SITK images
        '''
        self.fields = fields

    def forward(self, data):
        for field in self.fields:
            volume_list = []
            for elem in data[field]:
                volume = sitk.ReadImage(elem)
                volume_list.append(volume)
            data[field] = volume_list

        return data

    def backward(self, **kwargs):
        raise NotImplementedError


class FromSITKUint8ToSITKFloat32(object):
    def __init__(self, fields):
        '''
        FromSITKUint8ToSITKFloat32 forward method casts data to float32. the backward method does the contrary
        :param fields: fields of the dictionary whose content should be modified
        '''
        super(FromSITKUint8ToSITKFloat32, self).__init__()
        self.fields = fields

    def forward(self, data):
        filter = sitk.CastImageFilter()
        filter.SetOutputPixelType(sitk.sitkFloat32)
        for field in self.fields:
            for i in range(len(data[field])):
                data[field][i] = filter.Execute(data[field][i])

        return data

    def backward(self, data):
        filter = sitk.CastImageFilter()
        filter.SetOutputPixelType(sitk.sitkUInt8)
        for field in self.fields:
            for i in range(len(data[field])):
                data[field][i] = filter.Execute(data[field][i])

        return data


class FromSITKOriginalIntensitiesToRescaledIntensities(object):
    def __init__(self, fields, min_intensity=0., max_intensity=1.):
        '''
        FromSITKOriginalIntensitiesToRescaledIntensities forward method rescales data the correct intensity range.
        the backward method does the contrary.
        :param fields: fields of the dictionary whose content should be modified
        :param min_intensity: the lower end of the new intensity range
        :param max_intensity: the higher end of the new intensity range
        '''
        super(FromSITKOriginalIntensitiesToRescaledIntensities, self).__init__()

        self.fields = fields
        self.min = min_intensity
        self.max = max_intensity

    def forward(self, data):
        original_ranges_min = {}
        original_ranges_max = {}

        rescaling_fiter = sitk.RescaleIntensityImageFilter()
        rescaling_fiter.SetOutputMaximum(self.max)
        rescaling_fiter.SetOutputMinimum(self.min)

        min_max_filter = sitk.MinimumMaximumImageFilter()

        for field in self.fields:
            original_ranges_max[field] = []
            original_ranges_min[field] = []
            for i in range(len(data[field])):
                min_max_filter.Execute(data[field][i])

                original_ranges_max[field].append(min_max_filter.GetMaximum())
                original_ranges_min[field].append(min_max_filter.GetMinimum())

                data[field][i] = rescaling_fiter.Execute(data[field][i], self.min, self.max)

        data['original_ranges_min'] = original_ranges_min
        data['original_ranges_max'] = original_ranges_max

        return data

    def backward(self, data):
        original_ranges_min = data['original_ranges_min']
        original_ranges_max = data['original_ranges_max']
        for field in self.fields:
            for i in range(len(data[field])):
                rescaling_fiter = sitk.RescaleIntensityImageFilter()

                rescaling_fiter.SetOutputMaximum(original_ranges_max[field][i])
                rescaling_fiter.SetOutputMinimum(original_ranges_min[field][i])

                data[field][i] = rescaling_fiter.Execute(data[field][i])

        return data


class FromSITKOriginalResolutionToStandardResolution(object):
    def __init__(self, fields, resolution):
        '''
        FromSITKOriginalResolutionToStandardResolution makes all the volumes have the same resolution
        :param fields: fields of the dictionary whose content should be modified
        :param resolution: the new resolution of the data in three directions
        '''
        super(FromSITKOriginalResolutionToStandardResolution, self).__init__()
        self.fields = fields
        self.resolution = resolution

    def forward(self, data):
        original_spacings = {}

        for field in self.fields:
            original_spacings[field] = []
            for i in range(len(data[field])):
                factor = np.asarray(data[field][i].GetSpacing()) / np.asarray(self.resolution, dtype=float)
                new_size = np.asarray(data[field][i].GetSize() * factor, dtype=int)

                original_spacings[field].append(data[field][i].GetSpacing())

                resampler = sitk.ResampleImageFilter()
                resampler.SetReferenceImage(data[field][i])
                resampler.SetOutputSpacing(self.resolution)
                resampler.SetSize(new_size)

                data[field][i] = resampler.Execute(data[field][i])
        data['original_spacings'] = original_spacings

        return data

    def backward(self, data):
        original_spacings = data['original_spacings']

        for field in self.fields:
            for i in range(len(data[field])):
                factor = \
                    np.asarray(data[field][i].GetSpacing()) / np.asarray(original_spacings[field][i], dtype=float)
                new_size = np.asarray(data[field][i].GetSize() * factor, dtype=int)

                resampler = sitk.ResampleImageFilter()
                resampler.SetReferenceImage(data[field][i])
                resampler.SetOutputSpacing(original_spacings[field][i])
                resampler.SetSize(new_size)

                data[field][i] = resampler.Execute(data[field][i])

        return data


class FromSITKToNumpy(object):
    def __init__(self, fields):
        '''
        FromSITKToNumpy converts Sitk data in numpy arrays
        :param fields: fields of the dictionary whose content should be modified
        '''
        super(FromSITKToNumpy, self).__init__()
        self.fields = fields

    def forward(self, data):
        original_directions = {}
        original_origins = {}
        original_spacings = {}

        for field in self.fields:
            original_spacings[field] = []
            original_directions[field] = []
            original_origins[field] = []

            for i in range(len(data[field])):
                original_spacings[field].append(data[field][i].GetSpacing())
                original_directions[field].append(data[field][i].GetDirection())
                original_origins[field].append(data[field][i].GetOrigin())

                data[field][i] = np.transpose(sitk.GetArrayFromImage(data[field][i]).astype(dtype=np.float32), [2, 1, 0])

        data['original_spacings_NP'] = original_spacings
        data['original_directions_NP'] = original_directions
        data['original_origins_NP'] = original_origins

        return data

    def backward(self, data):
        original_spacings = data['original_spacings_NP']
        original_directions = data['original_directions_NP']
        original_origins = data['original_origins_NP']
        for field in self.fields:
            for i in range(len(data[field])):
                numpy_data = np.transpose(data[field][i], [2, 1, 0])

                data[field][i] = sitk.GetImageFromArray(numpy_data)
                data[field][i].SetDirection(original_directions[field][i])
                data[field][i].SetOrigin(original_origins[field][i])
                data[field][i].SetSpacing(original_spacings[field][i])

        return data


class FromNumpyOriginalSizeToStandardSize(object):
    def __init__(self, fields, size):
        '''
        FromNumpyOriginalSizeToStandardSize resizes data to predefined size. the backward method pads with zeros
        therefore it does not restore any lost information due to cropping
        :param fields: fields of the dictionary whose content should be modified
        :param size: desired image size in the three directions
        '''
        self.fields = fields
        self.size = np.asarray(size)

    def forward(self, data):
        pads = {}
        crops = {}
        original_sizes = {}
        for field in self.fields:
            pads[field] = []
            crops[field] = []
            original_sizes[field] = []
            for i in range(len(data[field])):
                original_sizes[field].append(np.asarray(data[field][i].shape))

                pad_amount = self.size - data[field][i].shape

                pad_before = np.floor(pad_amount / 2.).astype(int)
                pad_after = np.ceil(pad_amount / 2.).astype(int)

                pad_before[pad_before < 0] = 0
                pad_after[pad_after < 0] = 0

                pad_vec = ((pad_before[0], pad_after[0]), (pad_before[1], pad_after[1]), (pad_before[2], pad_after[2]))

                data_t = np.pad(data[field][i], pad_vec, mode='constant')

                crop_amount = data_t.shape - self.size

                crop_before = np.floor(crop_amount / 2.).astype(int)
                crop_after = np.ceil(crop_amount / 2.).astype(int)

                crop_before[crop_before < 0] = 0
                crop_after[crop_after < 0] = 0

                crop_vec = (
                    (crop_before[0], crop_after[0]),
                    (crop_before[1], crop_after[1]),
                    (crop_before[2], crop_after[2])
                )

                data_t = data_t[
                         crop_vec[0][0]:data_t.shape[0] - crop_vec[0][1],
                         crop_vec[1][0]:data_t.shape[1] - crop_vec[1][1],
                         crop_vec[2][0]:data_t.shape[2] - crop_vec[2][1]
                         ]

                crops[field].append(crop_vec)
                pads[field].append(pad_vec)

                assert np.all(data_t.shape == self.size)

                data[field][i] = data_t

        data['pads'] = pads
        data['crops'] = crops
        data['original_sizes_CP'] = original_sizes
        return data

    def backward(self, data):
        pads = data['pads']
        crops = data['crops']
        original_sizes = data['original_sizes_CP']
        for field in self.fields:
            for i in range(len(data[field])):
                data_t = np.pad(data[field][i], crops[field][i], mode='constant')
                data_t = data_t[
                         pads[field][i][0][0]:data_t.shape[0] - pads[field][i][0][1],
                         pads[field][i][1][0]:data_t.shape[1] - pads[field][i][1][1],
                         pads[field][i][2][0]:data_t.shape[2] - pads[field][i][2][1]
                         ]

                data[field][i] = data_t

                assert np.all(np.asarray(data[field][i].shape) == np.asarray(original_sizes[field][i]))

        return data


class FromListToNumpy5DArray(object):
    def __init__(self, fields):
        '''
        FromListToNumpy5DArray makes the data 5D and ready for inference
        :param fields: the fields that need to be tranformed to 5D
        '''

        self.fields = fields

    def forward(self, data):
        for field in self.fields:
            data[field] = np.stack(data[field])[..., np.newaxis]

        return data

    def backward(self, data):
        for field in self.fields:
            data_t = []
            for i in range(data[field].shape[0]):
                data_t.append(np.squeeze(data[field][i]))

            data[field] = data_t
        return data
