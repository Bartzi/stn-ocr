import numpy as np
import math

from ctc.ctc_loss import CTCLoss


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


a = np.random.rand(10, 10, 50).astype(np.float32)
b = np.array([1 for _ in range(10)], dtype=np.int32)

num_timesteps = 2
alphabet_size = 5

activations = np.array([0.1, 0.6, 0.1, 0.1, 0.1, 0.1, 0.1, 0.6, 0.1, 0.1], dtype=np.float32).reshape((num_timesteps, alphabet_size))
softmaxed_activations = softmax(activations)
expected_score = softmaxed_activations[0, 1] * softmaxed_activations[1, 2]

labels = np.array([1, 2], dtype=np.int32)

c = CTCLoss(num_timesteps, len(labels), 0)

np.testing.assert_allclose(
    math.exp(-c.calc_ctc_loss(labels, activations.reshape(num_timesteps, -1, alphabet_size))),
    expected_score,
    rtol=1e-6,
)

print("passed")
