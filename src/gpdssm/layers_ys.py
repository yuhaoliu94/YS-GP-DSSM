import numpy as np

from abc import ABC, abstractmethod
from copy import deepcopy

from src.gpdssm.utils import get_random_state, normalize_weights_special, get_sequential_mse, get_sequential_mnll
from src.gpdssm.distributions import Beta


class Layer(ABC):

    def __init__(self, dim: int, num_class: int = None, num_particle: int = 1000, num_rff: int = 50,
                 warm_start: int = 0, learning_rate: float = 0.001) -> None:
        self.C = num_class

        # assign after initializing structure
        self.next_layer = None
        self.prev_layer = None

        self.din = None
        self.function = None

        # scalar
        self.t = 0
        self.dim = dim
        self.dout = self.dim

        self.M = num_particle
        self.J = num_rff
        self.warm_start = warm_start
        self.learning_rate = learning_rate

    @abstractmethod
    def get_input_particles(self):
        raise NotImplementedError("Class must override get_input_particles")

    @abstractmethod
    def predict(self):
        raise NotImplementedError("Class must override predict")

    @abstractmethod
    def filter(self, *args):
        raise NotImplementedError("Class must override filter")

    @abstractmethod
    def update(self, *args):
        raise NotImplementedError("Class must override update")


# Yule Simon Layer
class YuleSimonLayer(Layer):
    def __init__(self, a0: np.array, b0: np.array, *args):
        super().__init__(*args)

        self.a0 = a0  # C
        self.b0 = b0  # C

        # state Z
        self.prev_particle_state = None
        self.current_particle_state = self.initialize_particle_state()  # M
        self.current_state = self.average_particle_state()  # C
        self.stored_particle_states = [self.current_particle_state]
        self.stored_states = [self.current_state]

        # rho
        self.current_particle_rho = self.initialize_particle_rho()  # M * C
        self.current_rho = self.average_particle_rho()  # C
        self.stored_rhos = [self.current_rho]

        # w, n, N
        self.current_particle_w = np.zeros((self.M, self.C))
        self.current_particle_n = np.zeros((self.M, self.C))
        self.current_particle_N = np.zeros((self.M, self.C))

        self.initialize_n()
        self.update_N()
        self.update_w()
        self.update_rho()

        self.particle_weights_for_next_layer = np.zeros((self.M, self.C))
        self.particle_weights_backwards = np.zeros((self.M, self.C))
        self.particle_weights_select = np.zeros((self.M, self.C))

        self.resample_indices = None

    def initialize_particle_state(self) -> np.ndarray:
        random_state = get_random_state()
        return random_state.choice(range(self.C), size=self.M, replace=True)

    def initialize_particle_rho(self) -> np.ndarray:
        beta_generator = Beta()
        return beta_generator.sample_multivariate(self.a0, self.b0, size=(self.M, self.C))  # M * C

    def initialize_n(self):
        self.current_particle_n[np.arange(self.M), self.current_particle_state] += 1

    def average_particle_state(self) -> np.ndarray:
        probability = np.zeros(self.C)
        for i in range(self.C):
            indices = self.current_particle_state == i
            probability[i] = np.mean(indices)

        return probability

    def average_particle_rho(self) -> np.ndarray:
        # probability_matrix = np.zeros((self.C, self.C))
        # for i in range(self.C):
        #     indices = self.current_particle_state == i
        #     probability_matrix[i, :] = np.average(self.current_particle_rho[indices, :], axis=0)
        probability_matrix = np.average(self.current_particle_rho, axis=0)
        return probability_matrix

    def get_input_particles(self):
        return

    def predict(self):
        self.prev_particle_state = deepcopy(self.current_particle_state)
        self.current_particle_state = np.ones((self.M, self.C)) * np.arange(self.C)

        transit_probability = self.get_transit_probability()
        self.particle_weights_for_next_layer = transit_probability  # M * C
        for i in range(self.C):
            self.next_layer[i].particle_weights_for_next_layer = transit_probability[:, i]

    def get_transit_probability(self):
        # z_prev = np.repeat(self.prev_particle_state[:, np.newaxis], self.C, axis=1)
        # z = self.current_particle_state
        rho = self.current_particle_rho
        n = self.current_particle_n
        N = self.current_particle_N

        stay_prob = n / (rho + n)
        # stay_prob_trunc = stay_prob * self.truncate_matrix_stay(self.prev_particle_state)

        Part_1 = rho / (rho + n)
        Part_2 = (rho + N)
        Part_2 *= self.truncate_matrix_change(self.prev_particle_state.astype(int))
        Part_2 = (Part_2.T / np.sum(Part_2, axis=1)).T
        change_prob = Part_1 * Part_2

        transit_prob = stay_prob + change_prob
        return transit_prob

    def truncate_matrix_stay(self, z):
        matrix = np.zeros((self.M, self.C))
        matrix[np.arange(self.M), z] = 1
        return matrix

    def truncate_matrix_change(self, z):
        matrix = np.zeros((self.M, self.C))
        matrix[np.arange(self.M), np.minimum(self.C-1, z+1)] = 1
        matrix[np.arange(self.M), np.maximum(0, z-1)] = 1
        matrix[np.arange(self.M), z] = 0
        return matrix

    def filter(self, *args):
        for i in range(self.C):
            self.particle_weights_backwards[:, i] = self.next_layer[i].particle_weights_for_prev_layer

        self.particle_weights_backwards = normalize_weights_special(self.particle_weights_backwards)
        self.particle_weights_select = self.particle_weights_backwards * self.particle_weights_for_next_layer
        argmax_indices = np.argmax(self.particle_weights_select, axis=1)

        self.current_particle_state = self.current_particle_state[np.arange(self.M), argmax_indices]
        self.stored_particle_states.append(deepcopy(self.current_particle_state))

        self.current_state = self.average_particle_state()
        self.stored_states.append(deepcopy(self.current_state))

    def update(self):
        self.update_w()
        self.t += 1

        self.update_rho()
        self.update_n()
        self.update_N()

        self.resample()

    def update_w(self):
        beta_generator = Beta()
        param_b = self.current_particle_state
        replicate_param = np.repeat(param_b[:, np.newaxis], self.C, axis=1)
        t = beta_generator.sample_array(self.current_particle_rho + 1, replicate_param + 1)
        self.current_particle_w -= np.log(t)

    def update_rho(self):
        beta_generator = Beta()
        param_a = self.a0 + self.t + 1
        replicate_param = np.repeat(param_a[np.newaxis, :], self.M, axis=0)
        self.current_particle_rho = beta_generator.sample_array(replicate_param, self.b0 + self.current_particle_w)
        self.current_rho = self.average_particle_rho()
        self.stored_rhos.append(self.current_rho)

    def update_n(self):
        stay = self.current_particle_state == self.prev_particle_state
        _class_same = self.current_particle_state[stay].astype(int)
        _class_diff = self.current_particle_state[~stay].astype(int)
        self.current_particle_n[stay, _class_same] += 1
        self.current_particle_n[~stay, :] = 0
        self.current_particle_n[~stay, _class_diff] = 1

    def update_N(self):
        self.current_particle_N[np.arange(self.M), self.current_particle_state.astype(int)] += 1

    def resample(self):
        random_state = get_random_state()
        resample_weights = self.particle_weights_backwards[np.arange(self.M), self.current_particle_state.astype(int)]
        _sum = np.sum(resample_weights)
        if _sum:
            resample_weights /= np.sum(resample_weights)
        else:
            resample_weights = np.ones(self.M) / self.M
        self.resample_indices = random_state.choice(range(self.M), size=self.M, p=resample_weights)

        self.current_particle_state = self.resample_particle(self.current_particle_state)
        self.current_particle_rho = self.resample_particle(self.current_particle_rho)
        self.current_particle_n = self.resample_particle(self.current_particle_n)
        self.current_particle_N = self.resample_particle(self.current_particle_N)
        self.current_particle_w = self.resample_particle(self.current_particle_w)

        stored_particle_states = np.array(self.stored_particle_states)  # t+1 * M
        stored_particle_states = stored_particle_states[:, self.resample_indices]
        self.stored_particle_states = list(stored_particle_states)

    def resample_particle(self, _array):
        if _array.ndim == 1:
            return _array[self.resample_indices]
        else:
            return _array[self.resample_indices, :]


# Transit Layer
class TransitLayer(Layer):
    def __init__(self, *args):
        super().__init__(*args)

        self.current_particle_state = None
        self.current_state = None
        self.stored_states = []

        self.particle_weights_for_prev_layer = np.zeros(self.M)
        self.particle_weights_for_next_layer = np.zeros(self.M)

    def initialize_particle_state(self, initial_particle_state):
        self.current_particle_state = initial_particle_state
        self.current_state = np.average(self.current_particle_state, axis=0)
        self.stored_states.append(deepcopy(self.current_state))

    def predict(self) -> None:
        self.current_particle_state = self.function.predict(self.get_input_particles())
        self.next_layer.particle_weights_for_next_layer = self.particle_weights_for_next_layer

    def filter(self) -> None:
        self.particle_weights_for_prev_layer = self.next_layer.particle_weights_for_prev_layer

    def update(self) -> None:
        self.t += 1

    @abstractmethod
    def get_input_particles(self) -> np.ndarray:
        raise NotImplementedError("Class must override get_input_particles")


class HiddenTransitLayer(TransitLayer):

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def get_input_particles(self) -> np.ndarray:
        input_particles = np.concatenate(
            (self.current_particle_state, self.prev_layer.current_particle_state), axis=1)

        return input_particles


class RootTransitLayer(TransitLayer):

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def get_input_particles(self) -> np.ndarray:
        return self.current_particle_state


# Non Transit Layer
class NonTransitLayer(Layer):
    def __init__(self, *args):
        super().__init__(*args)

        self.current_particle_state = None
        self.current_state = None
        self.stored_states = []

        self.particle_weights_for_prev_layer = np.zeros(self.M)
        self.particle_weights_for_next_layer = np.zeros(self.M)

    def get_input_particles(self) -> np.ndarray:
        return self.prev_layer.current_particle_state

    def predict(self) -> None:
        self.current_particle_state = self.function.predict(self.get_input_particles())
        self.next_layer.particle_weights_for_next_layer = self.particle_weights_for_next_layer

    def update(self) -> None:
        self.t += 1

    @abstractmethod
    def filter(self, *args):
        raise NotImplementedError("Class must override filter")


class HiddenNonTransitLayer(NonTransitLayer):

    def filter(self) -> None:
        self.particle_weights_for_prev_layer = self.next_layer.particle_weights_for_prev_layer


class ObservationLayer(NonTransitLayer):

    def __init__(self, *args) -> None:
        super().__init__(*args)

        self.y = None

    def predict(self) -> None:
        self.current_particle_state = self.function.predict(self.get_input_particles())

    def filter(self, y) -> None:
        self.y = y

        replicate_actual_realization = np.repeat(y[np.newaxis, :], self.M, axis=0)  # M * dim
        log_likelihood = self.function.cal_log_likelihood(replicate_actual_realization)  # M

        self.particle_weights_for_prev_layer = log_likelihood


class EnsembleObservationLayer(NonTransitLayer):

    def __init__(self, *args) -> None:
        super().__init__(*args)

        self.y = None

        self.y_log_likelihood_forward = 0.0
        self.stored_y_log_likelihood_forward = []

        self.mse = 0.0
        self.mnll = 0.0

    def get_prediction_particles(self) -> np.ndarray:
        self.C = len(self.prev_layer)
        ensemble_predict = []
        for i in range(self.C):
            ensemble_predict.append(deepcopy(self.prev_layer[i].current_particle_state))

        return np.array(ensemble_predict).T  # Dy * M * C

    def get_input_weights(self) -> np.ndarray:
        ensemble_weights = []
        for i in range(self.C):
            ensemble_weights.append(deepcopy(self.prev_layer[i].particle_weights_for_next_layer))

        return np.array(ensemble_weights).T  # M * C

    def get_backward_weights(self) -> np.ndarray:
        ensemble_weights = []
        for i in range(self.C):
            ensemble_weights.append(deepcopy(self.prev_layer[i].particle_weights_for_prev_layer))

        return np.array(ensemble_weights).T  # M * C

    def get_input_particles(self) -> np.ndarray:
        ensemble_input = []
        for i in range(self.C):
            ensemble_input.append(deepcopy(self.prev_layer[i].prev_layer.current_particle_state))

        return np.array(ensemble_input).T  # Dx * M * C

    def predict(self) -> None:
        weighted_prediction = self.get_prediction_particles() * self.get_input_weights()
        self.current_particle_state = np.sum(weighted_prediction, axis=2).T  # M * Dy
        self.current_state = np.average(self.current_particle_state, axis=0)
        self.stored_states.append(deepcopy(self.current_state))

        # backward_weights = self.get_backward_weights()
        # backward_weights = normalize_weights(backward_weights)
        # for i in range(self.C):
        #     self.prev_layer[i].particle_weights_for_prev_layer = backward_weights[:, i]

    def filter(self, y) -> None:
        self.y = y

        weighted_input = self.get_input_particles() * self.get_input_weights()
        ensemble_input = np.sum(weighted_input, axis=2).T  # M * Dx
        input_vector = np.average(ensemble_input, axis=0)
        cum_log_likelihood_forward = self.prev_layer[0].function.cal_y_log_likelihood_forward(input_vector, y)
        prev_cum_log_likelihood = self.prev_layer[0].function.cal_y_log_likelihood()
        self.y_log_likelihood_forward = cum_log_likelihood_forward - prev_cum_log_likelihood
        self.stored_y_log_likelihood_forward.append(self.y_log_likelihood_forward)

    def update(self) -> None:
        self.t += 1

        # mse
        self.mse = get_sequential_mse(self.mse, self.t - 1, self.y, self.current_state)

        # mnll
        self.mnll = get_sequential_mnll(self.mnll, self.t - 1, self.y_log_likelihood_forward)
