import pytest
import os
import numpy as np
from minisom import MiniSom, _build_iteration_indexes, fast_norm
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from numpy.testing import assert_array_equal
from numpy import zeros, ones, array, arange, random, array_equal, linalg, subtract, sqrt, max
from numpy.linalg import norm
import pickle

class TestMinisom():
    
    @pytest.fixture
    def som(self):
        ret = MiniSom(5, 5, 1)
        for i in range(5):
            for j in range(5):
                # checking weights normalization
                assert_almost_equal(1.0, linalg.norm(ret._weights[i, j]))
        ret._weights = zeros((5, 5, 1))  # fake weights
        ret._weights[2, 3] = 5.0
        ret._weights[1, 1] = 2.0
        return ret

    def test_decay_function(self, som):
        assert som._decay_function(1., 2., 3.) == 1./(1.+2./(3./2))

    def test_fast_norm(self):
        assert fast_norm(array([1, 3])) == sqrt(1+9)

    def test_euclidean_distance(self, som):
        x = zeros((1, 2))
        w = ones((2, 2, 2))
        d = som._euclidean_distance(x, w)
        assert_array_almost_equal(d, [[1.41421356, 1.41421356],
                                      [1.41421356, 1.41421356]])

    def test_cosine_distance(self, som):
        x = zeros((1, 2))
        w = ones((2, 2, 2))
        d = som._cosine_distance(x, w)
        assert_array_almost_equal(d, [[1., 1.],
                                      [1., 1.]])

    def test_manhattan_distance(self, som):
        x = zeros((1, 2))
        w = ones((2, 2, 2))
        d = som._manhattan_distance(x, w)
        assert_array_almost_equal(d, [[2., 2.],
                                      [2., 2.]])

    def test_chebyshev_distance(self, som):
        x = array([1, 3])
        w = ones((2, 2, 2))
        d = som._chebyshev_distance(x, w)
        assert_array_almost_equal(d, [[2., 2.],
                                      [2., 2.]])

    def test_check_input_len(self, som):
        with pytest.raises(ValueError):
            som.train_batch([[1, 2]], 1)

        with pytest.raises(ValueError):
            som.random_weights_init(array([[1, 2]]))

        with pytest.raises(ValueError):
            som._check_input_len(array([[1, 2]]))

        som._check_input_len(array([[1]]))
        som._check_input_len([[1]])

    def test_unavailable_neigh_function(self):
        with pytest.raises(ValueError):
            MiniSom(5, 5, 1, neighborhood_function='boooom')

    def test_unavailable_distance_function(self):
        with pytest.raises(ValueError):
            MiniSom(5, 5, 1, activation_distance='ridethewave')

    def test_gaussian(self, som):
        bell = som._gaussian((2, 2), 1)
        assert bell.max() == 1.0
        assert bell.argmax() == 12  # unravel(12) = (2,2)

    def test_mexican_hat(self, som):
        bell = som._mexican_hat((2, 2), 1)
        assert bell.max() == 1.0
        assert bell.argmax() == 12  # unravel(12) = (2,2)

    def test_bubble(self, som):
        bubble = som._bubble((2, 2), 1)
        assert bubble[2, 2] == 1
        assert sum(sum(bubble)) == 1

    def test_triangle(self, som):
        bubble = som._triangle((2, 2), 1)
        assert bubble[2, 2] == 1
        assert sum(sum(bubble)) == 1

    def test_win_map(self, som):
        winners = som.win_map([[5.0], [2.0]])
        assert winners[(2, 3)][0] == [5.0]
        assert winners[(1, 1)][0] == [2.0]

    def test_win_map_indices(self, som):
        winners = som.win_map([[5.0], [2.0]], return_indices=True)
        assert winners[(2, 3)] == [0]
        assert winners[(1, 1)] == [1]

    def test_labels_map(self, som):
        labels_map = som.labels_map([[5.0], [2.0]], ['a', 'b'])
        assert labels_map[(2, 3)]['a'] == 1
        assert labels_map[(1, 1)]['b'] == 1
        with pytest.raises(ValueError):
            som.labels_map([[5.0]], ['a', 'b'])

    def test_activation_reponse(self, som):
        response = som.activation_response([[5.0], [2.0]])
        assert response[2, 3] == 1
        assert response[1, 1] == 1

    def test_activate(self, som):
        assert som.activate(5.0).argmin() == 13.0  # unravel(13) = (2,3)

    def test_distance_from_weights(self, som):
        data = arange(-5, 5).reshape(-1, 1)
        weights = som._weights.reshape(-1, som._weights.shape[2])
        distances = som._distance_from_weights(data)
        for i in range(len(data)):
            for j in range(len(weights)):
                assert (distances[i][j] == norm(data[i] - weights[j]))

    def test_quantization_error(self, som):
        assert som.quantization_error([[5], [2]]) == 0.0
        assert som.quantization_error([[4], [1]]) == 1.0

    def test_topographic_error(self, som):
        # 5 will have bmu_1 in (2,3) and bmu_2 in (2, 4)
        # which are in the same neighborhood
        som._weights[2, 4] = 6.0
        # 15 will have bmu_1 in (4, 4) and bmu_2 in (0, 0)
        # which are not in the same neighborhood
        som._weights[4, 4] = 15.0
        som._weights[0, 0] = 14.
        assert som.topographic_error([[5]]) == 0.0
        assert som.topographic_error([[15]]) == 1.0

        som.topology = 'hexagonal'
        # 10 will have bmu_1 in (0, 4) and bmu_2 in (1, 3)
        # which are in the same neighborhood on a hexagonal grid
        som._weights[0, 4] = 10.0
        som._weights[1, 3] = 9.0
        # 3 will have bmu_1 in (2, 0) and bmu_2 in (1, 1)
        # which are in the same neighborhood on a hexagonal grid
        som._weights[2, 0] = 3.0
        assert som.topographic_error([[10]]) == 0.0
        assert som.topographic_error([[3]]) == 0.0
        # True for both hexagonal and rectangular grids
        assert som.topographic_error([[5]]) == 0.0
        assert som.topographic_error([[15]]) == 1.0
        som.topology = 'rectangular'

    def test_quantization(self, som):
        q = som.quantization(array([[4], [2]]))
        assert q[0] == 5.0
        assert q[1] == 2.0

    def test_random_seed(self):
        som1 = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
        som2 = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
        # same initialization
        assert_array_almost_equal(som1._weights, som2._weights)
        data = random.rand(100, 2)
        som1 = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
        som1.train_random(data, 10)
        som2 = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
        som2.train_random(data, 10)
        # same state after training
        assert_array_almost_equal(som1._weights, som2._weights)

    def test_train_batch(self):
        som = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
        data = array([[4, 2], [3, 1]])
        q1 = som.quantization_error(data)
        som.train(data, 10)
        assert q1 > som.quantization_error(data)

        data = array([[1, 5], [6, 7]])
        q1 = som.quantization_error(data)
        som.train_batch(data, 10, verbose=True)
        assert q1 > som.quantization_error(data)

    def test_train_random(self):
        som = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
        data = array([[4, 2], [3, 1]])
        q1 = som.quantization_error(data)
        som.train(data, 10, random_order=True)
        assert q1 > som.quantization_error(data)

        data = array([[1, 5], [6, 7]])
        q1 = som.quantization_error(data)
        som.train_random(data, 10, verbose=True)
        assert q1 > som.quantization_error(data)

    def test_train_use_epochs(self):
        som = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
        data = array([[4, 2], [3, 1]])
        q1 = som.quantization_error(data)
        som.train(data, 10, use_epochs=True)
        assert q1 > som.quantization_error(data)

    def test_use_epochs_variables(self):
        len_data = 100000
        num_epochs = 100
        random_gen = random.RandomState(1)
        iterations = _build_iteration_indexes(len_data, num_epochs,
                                              random_generator=random_gen,
                                              use_epochs=True)
        assert num_epochs*len_data == len(iterations)

        # checks whether all epochs share the same order of indexes
        first_epoch = iterations[0:len_data]
        for i in range(num_epochs):
            i_epoch = iterations[i*len_data:(i+1)*len_data]
            assert array_equal(first_epoch, i_epoch)

        # checks whether the decay_factor stays constant during one epoch
        # and that its values range from 0 to num_epochs-1
        decay_factors = []
        for t, iteration in enumerate(iterations):
            decay_factor = int(t / len_data)
            decay_factors.append(decay_factor)
        for i in range(num_epochs):
            decay_factors_i_epoch = decay_factors[i*len_data:(i+1)*len_data]
            assert decay_factors_i_epoch == [i]*len_data

    def test_random_weights_init(self):
        som = MiniSom(2, 2, 2, random_seed=1)
        som.random_weights_init(array([[1.0, .0]]))
        for w in som._weights:
            assert_array_equal(w[0], array([1.0, .0]))

    def test_pca_weights_init(self):
        som = MiniSom(2, 2, 2)
        som.pca_weights_init(array([[1.,  0.], [0., 1.], [1., 0.], [0., 1.]]))
        expected = array([[[-1.41421356,  0.],
                           [0.,  1.41421356]],
                          [[0., -1.41421356],
                           [1.41421356,  0.]]])
        assert_array_almost_equal(som._weights, expected)

    def test_distance_map(self):
        som = MiniSom(2, 2, 2, random_seed=1)
        som._weights = array([[[1.,  0.], [0., 1.]], [[1., 0.], [0., 1.]]])
        assert_array_equal(som.distance_map(), array([[1., 1.], [1., 1.]]))

        som = MiniSom(2, 2, 2, topology='hexagonal', random_seed=1)
        som._weights = array([[[1.,  0.], [0., 1.]], [[1., 0.], [0., 1.]]])
        assert_array_equal(som.distance_map(), array([[.5, 1.], [1., .5]]))

        som = MiniSom(3, 3, 1, random_seed=1)
        som._weights = array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
        dist = array([[2/3, 3/5, 2/3], [3/5, 4/8, 3/5], [2/3, 3/5, 2/3]])
        assert_array_equal(som.distance_map(scaling='mean'), dist/np.max(dist))

        with pytest.raises(ValueError):
            som.distance_map(scaling='puppies')

    def test_pickling(self, som):
        with open('som.p', 'wb') as outfile:
            pickle.dump(som, outfile)
        with open('som.p', 'rb') as infile:
            pickle.load(infile)
        os.remove('som.p')

    def test_callable_activation_distance(self):
        def euclidean(x, w):
            return linalg.norm(subtract(x, w), axis=-1)

        data = random.rand(100, 2)
        som1 = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5,
                       activation_distance=euclidean, random_seed=1)
        som1.train_random(data, 10)
        som2 = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
        som2.train_random(data, 10)
        # same state after training
        assert_array_almost_equal(som1._weights, som2._weights)
