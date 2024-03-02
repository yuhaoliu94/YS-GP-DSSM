import numpy as np

from abc import ABC
from time import time
from copy import deepcopy
from src.gpdssm.distributions import Normal
from src.gpdssm.utils import import_dataset, get_mse, get_mnll, get_svd_representation_list
from src.gpdssm.layers_ys import (YuleSimonLayer, HiddenTransitLayer, RootTransitLayer, ObservationLayer,
                                  EnsembleObservationLayer, HiddenNonTransitLayer)


class ModelLayer(ABC):

    def __init__(self, model_list: list):
        # self.model_list = model_list
        self.layers_list = [model.layers for model in model_list]  # C * num_all_layer
        self.hidden_layers_list = [model.hidden_layers for model in model_list]
        self.observation_layer_list = [model.observation_layer for model in model_list]

        self.num_all_layer = len(self.layers_list[0])
        self.num_hidden_layer = len(self.hidden_layers_list[0])

        self.t = 0
        self.C = len(self.layers_list)

        self.M = self.layers_list[0][0].M

        self.prev_layer = None
        self.next_layer = None

        self.resample_indices = None
        self.particle_weights_select = None

        self.initialize_particle_state()

    def initialize_particle_state(self):
        for i in range(self.num_hidden_layer):
            dim = self.hidden_layers_list[0][i].dim
            normal_generator = Normal()
            initial_particle_state = normal_generator.sample_univariate(0, 1, (self.M, dim))
            for c in range(self.C):
                self.hidden_layers_list[c][i].initialize_particle_state(initial_particle_state)

    def get_ys_layer(self):
        return self.layers_list[0][0].prev_layer

    def get_obs_layer(self):
        return self.layers_list[0][-1].next_layer

    def get_resample_indices(self):
        ys_layer = self.get_ys_layer()
        self.resample_indices = ys_layer.resample_indices

    def get_particle_weights_select(self):
        ys_layer = self.get_ys_layer()
        self.particle_weights_select = ys_layer.particle_weights_select

    def predict(self):
        for c in range(self.C):
            for i in range(self.num_all_layer):
                self.layers_list[c][i].predict()

    def filter(self, y):
        for c in range(self.C):
            self.observation_layer_list[c].filter(y)
        for c in range(self.C):
            for i in reversed(range(self.num_hidden_layer)):
                self.hidden_layers_list[c][i].filter()

    def filter_states(self):
        self.get_particle_weights_select()
        argmax_indices = np.argmax(self.particle_weights_select, axis=1)
        for i in range(self.num_hidden_layer):
            ensemble_particles = self.get_ensemble_particles(i)  # C * M * Dx
            current_particle_state = ensemble_particles[argmax_indices, np.arange(self.M), :]  # M * Dx
            current_state = np.average(current_particle_state, axis=0)

            for c in range(self.C):
                self.hidden_layers_list[c][i].current_particle_state = deepcopy(current_particle_state)
                self.hidden_layers_list[c][i].current_state = deepcopy(current_state)
                self.hidden_layers_list[c][i].stored_states.append(deepcopy(current_state))

    def get_ensemble_particles(self, idx_layer) -> np.ndarray:
        ensemble_input = []
        for c in range(self.C):
            ensemble_input.append(deepcopy(self.hidden_layers_list[c][idx_layer].current_particle_state))

        return np.array(ensemble_input)  # C * M * Dx

    def update(self):
        for c in range(self.C):
            for i in range(self.num_all_layer):
                self.layers_list[c][i].update()

        self.resample()

        self.t += 1

    def resample(self):
        self.get_resample_indices()
        for i in range(self.num_hidden_layer):
            for c in range(self.C):
                resampled_array = self.resample_particle(self.hidden_layers_list[c][i].current_particle_state)
                self.hidden_layers_list[c][i].current_particle_state = resampled_array

    def resample_particle(self, _array):
        if _array.ndim == 1:
            return _array[self.resample_indices]
        else:
            return _array[self.resample_indices, :]


class YuleSimonModel(ABC):
    def __init__(self, data_name: str, data_fold: int, model_list: list,
                 num_rff: int, num_particle: int, a0: np.ndarray, b0: np.ndarray,
                 display_step: int, warm_start: int, learning_rate: float, num_train_cycle: int):

        self.t = 0
        self.M = num_particle
        self.J = num_rff
        self.C = len(model_list)

        self.a0 = a0
        self.b0 = b0

        self.display_step = display_step
        self.warm_start = warm_start
        self.learning_rate = learning_rate
        self.num_train_cycle = num_train_cycle

        # data
        self.data_name = data_name
        self.data_fold = data_fold
        self.data = import_dataset(self.data_name, self.data_fold)
        self.Dy = self.data.Dy

        # layers
        self.num_hidden_layer = 2
        self.num_all_layer = 3

        self.ys_layer = None
        self.model_layer = ModelLayer(model_list)
        self.ensemble_observation_layer = None

        self.layers = []
        self.initialize_structure()

        self.time = []

    def initialize_structure(self):
        constant_param = (self.C, self.M, self.J, self.warm_start, self.learning_rate)

        self.ys_layer = YuleSimonLayer(self.a0, self.b0, 1, *constant_param)
        self.ensemble_observation_layer = EnsembleObservationLayer(self.Dy, *constant_param)

        self.layers = [self.ys_layer, self.model_layer, self.ensemble_observation_layer]

        self.ys_layer.next_layer = [self.model_layer.layers_list[i][0] for i in range(self.C)]
        for i in range(self.C):
            self.model_layer.layers_list[i][0].prev_layer = self.ys_layer

        self.ensemble_observation_layer.prev_layer = [self.model_layer.layers_list[i][-1] for i in range(self.C)]
        for i in range(self.C):
            self.model_layer.layers_list[i][-1].next_layer = self.ensemble_observation_layer

    def predict(self):
        for i in range(self.num_all_layer):
            self.layers[i].predict()

    def filter(self):
        y = self.data.Y[self.t, :]
        for i in reversed(range(self.num_all_layer)):
            self.layers[i].filter(y)

    def update(self):
        for i in range(self.num_all_layer):
            self.layers[i].update()

        self.t += 1

    def learn(self) -> None:
        T = self.data.Num_Observations
        self.time.append(time())

        for n in range(self.num_train_cycle):

            print(">>> train cycle %d" % (n + 1), end="\n")

            for i in range(T):

                self.predict()
                self.filter()
                self.update()

                if i == 0 or i == T - 1 or (i + 1) % self.display_step == 0:
                    self.log_print(n + 1)

    def log_print(self, cycle: int) -> None:
        cum_mse = np.round(self.ensemble_observation_layer.mse, 4)
        cum_mnll = np.round(self.ensemble_observation_layer.mnll, 4)

        current_time = time()
        self.time.append(current_time)
        minutes = np.round((current_time - self.time[0]) / 60, 2)

        print(">>> cycle=%s, t=%s, mse=%s, mnll=%s, time=%s minutes" %
              (cycle, self.t, cum_mse, cum_mnll, minutes), end="\n")
