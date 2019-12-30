import unittest

import hypothesis.strategies as st
import numpy as np
from hypothesis import given
from hypothesis import settings

import fmnist.learning.arch.base


class FlatTo3DImageLayerTest(unittest.TestCase):
    MAX_ARMS = 5000

    @staticmethod
    def channels(expand_as_rgb: bool):
        return 3 if expand_as_rgb else 1

    @given(st.integers(min_value=1, max_value=1000),
           st.booleans())
    @settings(max_examples=10, deadline=None)
    def test_init_set_config_correctly(self, image_size: int, expand_as_rgb: bool):
        layer = fmnist.learning.arch.base.FlatTo3DImageLayer(image_2d_dimensions=(image_size, image_size),
                                                             expand_as_rgb=expand_as_rgb)
        self.assertEqual((image_size, image_size), layer.image_2d_dimensions)
        self.assertEqual(self.channels(expand_as_rgb), layer.channels)

    @given(st.integers(min_value=1, max_value=1000),
           st.booleans(),
           st.integers(min_value=1, max_value=100))
    @settings(max_examples=10, deadline=None)
    def test_compute_output_shape(self, image_size: int, expand_as_rgb: bool, batch_size: int):
        layer = fmnist.learning.arch.base.FlatTo3DImageLayer(image_2d_dimensions=(image_size, image_size),
                                                             expand_as_rgb=expand_as_rgb)

        _input = np.random.rand(batch_size, image_size * image_size)
        expected = (_input.shape[0], image_size, image_size, self.channels(expand_as_rgb))
        self.assertEqual(expected, layer.compute_output_shape(_input.shape))

    @given(st.integers(min_value=1, max_value=1000),
           st.booleans(),
           st.integers(min_value=1, max_value=100))
    @settings(max_examples=10, deadline=None)
    def test_layer_converts_image_vectors_to_3d_images(self, image_size: int, expand_as_rgb: bool, batch_size: int):
        _input = np.random.rand(batch_size, image_size * image_size)
        layer = fmnist.learning.arch.base.FlatTo3DImageLayer(image_2d_dimensions=(image_size, image_size),
                                                             expand_as_rgb=expand_as_rgb)
        layer.build(input_shape=_input.shape)

        output = layer.call(_input)
        expected = (batch_size, image_size, image_size, self.channels(expand_as_rgb))

        self.assertEqual(expected, output.shape)
        for i in range(self.channels(expand_as_rgb)):
            self.assertTrue(np.array_equal(_input, np.reshape(output[:, :, :, i],
                                                              newshape=(-1, image_size * image_size))))
