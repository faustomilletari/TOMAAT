import SimpleITK as sitk
import vtk
import numpy as np
import os

from vtk.util.numpy_support import numpy_to_vtk


'''
NOTE: The transforms that are added to the data during inference must be IDENTICAL to 
those used during training on the original data, at least for what concerns the forward transform step
'''


class FromITKFormatFilenameToSITK(object):
    def __init__(self, fields):
        '''
        FromITKFormatFilenameToSITK loads ITK compatible files
        :param fields: fields of the dictionary whose content should be replaced by SITK images
        '''
        self.fields = fields

    def __call__(self, data):
        for field in self.fields:
            volume_list = []
            for elem in data[field]:
                volume = sitk.ReadImage(elem)
                volume_list.append(volume)
                os.remove(elem)
            data[field] = volume_list

        return data


class FromSITKUint8ToSITKFloat32(object):
    def __init__(self, fields):
        '''
        FromSITKUint8ToSITKFloat32 casts data to float32.
        :param fields: fields of the dictionary whose content should be modified
        '''
        super(FromSITKUint8ToSITKFloat32, self).__init__()
        self.fields = fields

    def __call__(self, data):
        filter = sitk.CastImageFilter()
        filter.SetOutputPixelType(sitk.sitkFloat32)
        for field in self.fields:
            for i in range(len(data[field])):
                data[field][i] = filter.Execute(data[field][i])

        return data


class FromSITKFloat32ToSITKUint8(object):
    def __init__(self, fields):
        '''
        FromSITKFloat32ToSITKUint8 method casts data back to uint8.
        :param fields: fields of the dictionary whose content should be modified
        '''
        super(FromSITKFloat32ToSITKUint8, self).__init__()
        self.fields = fields

    def __call__(self, data):
        filter = sitk.CastImageFilter()
        filter.SetOutputPixelType(sitk.sitkUInt8)
        for field in self.fields:
            for i in range(len(data[field])):
                data[field][i] = filter.Execute(data[field][i])

        return data


class FromSITKOriginalIntensitiesToRescaledIntensities(object):
    def __init__(self,
                 fields,
                 min_intensity=0.,
                 max_intensity=1.,
                 field_original_ranges_min='original_ranges_min',
                 field_original_ranges_max='original_ranges_max'
                 ):
        '''
        FromSITKOriginalIntensitiesToRescaledIntensities method rescales data the correct intensity range.
        :param fields: fields of the dictionary whose content should be modified
        :param min_intensity: the lower end of the new intensity range
        :param max_intensity: the higher end of the new intensity range
        :param field_original_ranges_min: field to use in data dictionary to store original intensity minima
        :param field_original_ranges_max: field to use in data dictionary to store original intensity maxima
        '''
        super(FromSITKOriginalIntensitiesToRescaledIntensities, self).__init__()

        self.fields = fields
        self.min = min_intensity
        self.max = max_intensity
        self.field_original_ranges_min = field_original_ranges_min
        self.field_original_ranges_max = field_original_ranges_max

    def __call__(self, data):
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

        data[self.field_original_ranges_min] = original_ranges_min
        data[self.field_original_ranges_max] = original_ranges_max

        return data


class FromSITKRescaledIntensitiesToOriginalIntensities(object):
    def __init__(self,
                 fields,
                 field_original_ranges_min='original_ranges_min',
                 field_original_ranges_max='original_ranges_max'
                 ):
        '''
        FromSITKRescaledIntensitiesToOriginalIntensities rescales data the original intensity range.
        :param fields: fields of the dictionary whose content should be modified
        :param field_original_ranges_min: field in the data dictionary containing the minima of the original volumes
        :param field_original_ranges_max: field in the data dictionary containing the maxima of the original volumes
        '''
        super(FromSITKRescaledIntensitiesToOriginalIntensities, self).__init__()

        self.fields = fields
        self.field_original_ranges_min = field_original_ranges_min
        self.field_original_ranges_max = field_original_ranges_max

    def __call__(self, data):
        original_ranges_min = data[self.field_original_ranges_min]
        original_ranges_max = data[self.field_original_ranges_max]
        for field in self.fields:
            for i in range(len(data[field])):
                rescaling_fiter = sitk.RescaleIntensityImageFilter()

                rescaling_fiter.SetOutputMaximum(original_ranges_max[field][i])
                rescaling_fiter.SetOutputMinimum(original_ranges_min[field][i])

                data[field][i] = rescaling_fiter.Execute(data[field][i])

        return data


class FromSITKOriginalResolutionToStandardResolution(object):
    def __init__(self, fields, resolution, field_original_spacings='original_spacings', field_spacing_metric=None):
        '''
        FromSITKOriginalResolutionToStandardResolution makes all the volumes have the same resolution
        :param fields: fields of the dictionary whose content should be modified
        :param resolution: the new resolution of the data in three directions
        :param field_original_spacings: field to use for the data dictionary to store original resolutions
        '''
        super(FromSITKOriginalResolutionToStandardResolution, self).__init__()
        self.fields = fields
        self.resolution = resolution
        self.field_original_spacings = field_original_spacings
        self.field_spacing_metric = field_spacing_metric

    def __call__(self, data):
        original_spacings = {}

        for field in self.fields:
            original_spacings[field] = []

            for i in range(len(data[field])):

                if self.field_spacing_metric is not None:
                    if data[self.field_spacing_metric][i] == 'meters':
                        resolution = np.asarray(self.resolution) / 1000.
                    elif data[self.field_spacing_metric][i] == 'millimeters':
                        resolution = np.asarray(self.resolution)
                else:
                    resolution = np.asarray(self.resolution)

                factor = np.asarray(data[field][i].GetSpacing()) / np.asarray(resolution, dtype=float)
                new_size = np.asarray(data[field][i].GetSize() * factor, dtype=int)

                original_spacings[field].append(data[field][i].GetSpacing())

                resampler = sitk.ResampleImageFilter()
                resampler.SetReferenceImage(data[field][i])
                resampler.SetOutputSpacing(resolution)
                resampler.SetSize(new_size)

                data[field][i] = resampler.Execute(data[field][i])
        data[self.field_original_spacings] = original_spacings

        return data


class FromSITKStandardResolutionToOriginalResolution(object):
    def __init__(self, fields, field_original_spacings='original_spacings'):
        '''
        FromSITKStandardResolutionToOriginalResolution makes all the volumes return to their original resolution
        :param fields: fields of the dictionary whose content should be modified
        :param field_original_spacings: the fields of the data dictionary where the original resolutions are stored
        '''
        super(FromSITKStandardResolutionToOriginalResolution, self).__init__()
        self.fields = fields
        self.field_original_spacings = field_original_spacings

    def __call__(self, data):
        original_spacings = data[self.field_original_spacings]

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
    def __init__(self,
                 fields,
                 field_original_spacing='original_spacings_NP',
                 field_original_direction='original_directions_NP',
                 field_original_origins='original_origins_NP'
                 ):
        '''
        FromSITKToNumpy converts SITK data in numpy arrays
        :param fields: fields of the dictionary whose content should be modified
        :param field_original_spacing: field data dictionary used to store original spacing of sitk volumes
        :param field_original_direction: field data dictionary used to store original direction of sitk volumes
        :param field_original_origins: field data dictionary used to store original origin coordinate of sitk volumes
        '''
        super(FromSITKToNumpy, self).__init__()
        self.fields = fields
        self.field_original_spacing = field_original_spacing
        self.field_original_direction = field_original_direction
        self.field_original_origins = field_original_origins

    def __call__(self, data):
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

        data[self.field_original_spacing] = original_spacings
        data[self.field_original_direction] = original_directions
        data[self.field_original_origins] = original_origins

        return data


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

                data[field][i] = sitk.GetImageFromArray(numpy_data)
                data[field][i].SetDirection(original_directions[field][i])
                data[field][i].SetOrigin(original_origins[field][i])
                data[field][i].SetSpacing(original_spacings[field][i])

        return data


class FromNumpyOriginalSizeToStandardSize(object):
    def __init__(self,
                 fields,
                 size,
                 field_pads='pads_std_size',
                 field_crops="crops_std_size",
                 field_original_sizes='original_sizes_std_size'
                 ):
        '''
        FromNumpyOriginalSizeToStandardSize resizes data to predefined size.
        :param fields: fields of the dictionary whose content should be modified
        :param size: desired image size in the three directions
        :param field_pads: field of data dictionary to use to store paddings used in this transform
        :param field_crops: field of data dictionary to use to store crops used in this transform
        :param field_original_sizes: field of data dictionary to use to store sizes used in this transform
        '''
        self.fields = fields
        self.size = np.asarray(size)

        self.field_pads = field_pads
        self.field_crops = field_crops
        self.field_original_sizes = field_original_sizes

    def __call__(self, data):
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

        data[self.field_pads] = pads
        data[self.field_crops] = crops
        data[self.field_original_sizes] = original_sizes
        return data


class FromNumpyStandardSizeToOriginalSize(object):
    def __init__(self,
                 fields,
                 field_pads='pads_std_size',
                 field_crops="crops_std_size",
                 field_original_sizes='original_sizes_std_size'
                 ):
        '''
        FromNumpyOriginalSizeToStandardSize resizes data to original size. This method pads with zeros
        therefore it does not restore any lost information due to cropping
        :param fields: fields of the dictionary whose content should be modified
        :param field_pads: field of data dictionary to use to read paddings used in this transform
        :param field_crops: field of data dictionary to use to read crops used in this transform
        :param field_original_sizes: field of data dictionary to use to read sizes used in this transform
        '''
        self.fields = fields

        self.field_pads = field_pads
        self.field_crops = field_crops
        self.field_original_sizes = field_original_sizes

    def __call__(self, data):
        pads = data[self.field_pads]
        crops = data[self.field_crops]
        original_sizes = data[self.field_original_sizes]
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
        :param fields: the fields that need to be transformed to 5D
        '''

        self.fields = fields

    def __call__(self, data):
        for field in self.fields:
            data[field] = np.stack(data[field])[..., np.newaxis]

        return data


class FromNumpy5DArrayToList(object):
    def __init__(self, fields):
        '''
        FromNumpy5DArrayToList makes the data a list of numpy tensors
        :param fields: the fields that need to be transformed to 5D
        '''

        self.fields = fields

    def __call__(self, data):
        for field in self.fields:
            data_t = []
            for i in range(data[field].shape[0]):
                data_t.append(np.squeeze(data[field][i]))

            data[field] = data_t
        return data


class FromLabelVolumeToVTKMesh(object):
    mesh_reduction_percentage = 0.90

    def __init__(
            self,
            label_filed,
            mesh_field,
            return_VTK_field='return_VTK',
            spacings_field='original_spacings_NP',
            sizes_field='original_sizes_std_size',
            origins_field='original_origins_NP',
            directions_field='original_directions_NP',
            convert_to_ras_field='RAS'
    ):

        self.label_filed = label_filed
        self.mesh_field = mesh_field
        self.return_VTK_field = return_VTK_field
        self.spacings_field = spacings_field
        self.sizes_field = sizes_field
        self.origins_field = origins_field
        self.directions_field = directions_field
        self.convert_to_ras_field = convert_to_ras_field

    def __call__(self, data):
        meshes = []

        for label, spacing, origin, size, direction, return_VTK, convert_to_ras in zip(
                data[self.label_filed],
                data[self.spacings_field][self.label_filed],
                data[self.origins_field][self.label_filed],
                data[self.sizes_field][self.label_filed],
                data[self.directions_field][self.label_filed],
                data[self.return_VTK_field],
                data[self.convert_to_ras_field],
        ):
            if return_VTK == "False":
                continue

            label = np.transpose(np.copy(label), [2, 1, 0])

            vtk_voxelmap = vtk.vtkImageData()
            vtk_voxelmap.SetSpacing(spacing[0], spacing[1], spacing[2])
            vtk_voxelmap.SetOrigin(origin[0], origin[1], origin[2])
            vtk_voxelmap.SetDimensions(size[0], size[1], size[2])
            vtk_voxelmap.SetScalarType(vtk.VTK_FLOAT)

            vtk_data_array = numpy_to_vtk(num_array=label.ravel(), deep=True, array_type=vtk.VTK_FLOAT)

            points = vtk_voxelmap.GetPointData()
            points.SetScalars(vtk_data_array)

            contour = vtk.vtkDiscreteMarchingCubes()
            contour.SetInput(vtk_voxelmap)
            contour.ComputeNormalsOn()
            contour.GenerateValues(1, 1, 1)
            contour.Update()

            polydata = vtk.vtkPolyData()
            polydata.DeepCopy(contour.GetOutput())

            smoother = vtk.vtkSmoothPolyDataFilter()
            smoother.SetInput(polydata)
            smoother.SetNumberOfIterations(150)
            smoother.SetFeatureAngle(60)
            smoother.SetRelaxationFactor(0.05)
            smoother.FeatureEdgeSmoothingOff()
            smoother.Update()

            transform = vtk.vtkTransform()
            if convert_to_ras == 'True':
                mat = np.asarray(direction).reshape([3, 3])
                conversion_mat = np.asarray([-1, 0, 0, 0, -1, 0, 0, 0, 1]).reshape([3, 3])
                direction = np.matmul(conversion_mat, mat).flatten()

            dir_homo = np.asarray([direction[0], direction[1], direction[2], 0.,
                                   direction[3], direction[4], direction[5], 0.,
                                   direction[6], direction[7], direction[8], 0.,
                                   0., 0., 0., 1.]).reshape([4, 4])

            transform.SetMatrix(dir_homo.flatten())

            transformFilter = vtk.vtkTransformPolyDataFilter()
            transformFilter.SetTransform(transform)
            transformFilter.SetInputConnection(smoother.GetOutputPort())
            transformFilter.Update()

            mesh = vtk.vtkPolyData()

            mesh.ShallowCopy(transformFilter.GetOutput())

            decimator = vtk.vtkDecimatePro()
            decimator.SetInput(mesh)
            decimator.SetTargetReduction(self.mesh_reduction_percentage)
            decimator.SetPreserveTopology(1)
            decimator.Update()

            clean_poly_data_filter = vtk.vtkCleanPolyData()
            clean_poly_data_filter.SetInput(decimator.GetOutput())
            clean_poly_data_filter.Update()

            meshes.append(clean_poly_data_filter.GetOutput())

        data[self.mesh_field] = meshes

        return data


class ThresholdNumpy(object):
    def __init__(self, image_field, threshold_field):
        self.image_field = image_field
        self.threshold_field = threshold_field

    def __call__(self, data):
        data[self.image_field] = (data[self.image_field] >= data[self.threshold_field]).astype(np.float32)

        return data
