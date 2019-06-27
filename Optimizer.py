import numpy as np


class GradientDescent:
    def __init__(self, learning_rate=0.99):
        self.learning_rate = learning_rate

    def __call__(self, params, grads):

        for param_key in params:
            params[param_key] -= self.learning_rate * grads[param_key]

        return params

    def pull_state(self):
        return {}

    def push_state(self,state):
        pass



class AdamOptimizer:
    def __init__(self, gradient_clip=False, learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, decay=0.):
        self.iterations = 0
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.decay = decay
        self.epsilon = epsilon
        self.initial_decay = decay
        self.gradient_clip = gradient_clip

    def __call__(self, params, grads):

        if self.gradient_clip:
            norm = np.sqrt(sum([np.sum(np.square(grads[g])) for g in grads]))
            for g_key in grads:
                grads[g_key] = np.clip(grads[g_key],0.,norm)

        learning_rate = self.learning_rate
        if self.initial_decay > 0:
            learning_rate *= (1. / (1. + self.decay * self.iterations))

        t = self.iterations + 1
        lr_t = learning_rate * (np.sqrt(1. - np.power(self.beta_2, t)) /
                     (1. - np.power(self.beta_1, t)))

        if not hasattr(self, 'ms'):
            self.M = [np.zeros(params[p].shape) for p in params]
            self.R = [np.zeros(params[p].shape) for p in params]

        ret = {}

        i_counter = 0
        for param_key in params:
            M_hat = (self.beta_1 * self.M[i_counter]) + (1. - self.beta_1) * grads[param_key]
            R_hat = (self.beta_2 * self.R[i_counter]) + (1. - self.beta_2) * np.square(grads[param_key])

            ret[param_key] = params[param_key] - lr_t * M_hat / (np.sqrt(R_hat) + self.epsilon)

            self.M[i_counter] = M_hat
            self.R[i_counter] = R_hat

            i_counter += 1

        self.iterations += 1

        return ret

    def pull_state(self):
        return {"iterations":self.iterations, "M":self.M, "R":self.R}

    def push_state(self, state):
        assert len(state) == 3, "Unexpected number of state keys: "+ str(list(state.keys()))
        self.iterations, self.M, self.R = state["iterations"], state["M"], state["R"]



