#My own code
#Implemented using Algorithm 4 on p.101 in
# "Machine Learning with Neural Netorks" 2022 by Bernhard Mehlig.


import numpy as np

def initialize_weights(M1, M2, seed=42):
    rng = np.random.default_rng(42)
    #Input to layer 1
    w_1i = rng.normal(0, 0.1, (M1,2))
    #Layer 1 to layer 2
    w_21 = rng.normal(0, 0.1, (M2,M1))
    #Layer 2 to output
    w_o3 = rng.normal(0, 0.1, (M2,1))
    return [w_1i, w_21, w_o3]

def initialize_thresholds(M1, M2):
    theta_1 = np.zeros((M1,1))
    theta_2 = np.zeros((M2,1))
    theta_o = np.zeros((1,))
    return [theta_1, theta_2, theta_o]

def forward_propagation(w, theta, x):
    x = x.reshape(-1,1)
    v_1 = np.tanh(w[0] @ x - theta[0])
    v_2 = np.tanh(w[1] @ v_1 - theta[1])
    o = np.tanh(w[2].T @ v_2 - theta[2])
    return v_1, v_2, o

def classification_error(w, theta, x_validation, t_validation):
    prediction = []
    for xi, target in zip(x_validation, t_validation):
        _, _, o = forward_propagation(w, theta, xi)
        prediction.append(np.sign(o))
    prediction = np.array(prediction).flatten()
    C = 0.5 * np.mean(np.abs(prediction - t_validation))
    return C

def backward_propagation(w, v_1, v_2, o, target):
    delta_o = (o - target) * (1 - o**2)
    delta_2 = (w[2] * delta_o) * (1 - v_2 ** 2)
    delta_1 = (w[1].T @ delta_2) * (1 - v_1**2)
    return [delta_1, delta_2, delta_o]

def update_weights(w, delta, v_1, v_2, x, eta):
    w[0] -= eta * delta[0] @ x.reshape(1,-1)
    w[1] -= eta * delta[1] @ v_1.T
    w[2] -= eta * (v_2 * delta[2])
    return w

def update_thresholds(delta, eta, theta):
    theta[0] += eta * delta[0]
    theta[1] += eta * delta[1]
    theta[2] += eta * delta[2].item()
    return theta


m_1, m_2 = 8,8
eta = 0.01
epochs = 350
w = initialize_weights(m_1, m_2)
theta = initialize_thresholds(m_1, m_2)


training_data = np.loadtxt("training_set.csv", delimiter=",")
validation_data = np.loadtxt("validation_set.csv", delimiter=",")

x_training, t_training = training_data[:, :2], training_data[:,2]
x_validation, t_validation = validation_data[:, :2], validation_data[:, 2]



for epoch in range(epochs):
    for x, t in zip(x_training, t_training):
        v_1, v_2, o = forward_propagation(w, theta, x)
        delta = backward_propagation(w, v_1, v_2, o, t)
        w = update_weights(w, delta, v_1, v_2, x, eta)
        theta = update_thresholds(delta, eta, theta)

    C = classification_error(w, theta, x_validation, t_validation)
    print(f"Validation error = {C:.3f}")

np.savetxt("w1.csv",w[0], delimiter=",")
np.savetxt("w2.csv",w[1],delimiter=",")
np.savetxt("w3.csv", w[2].reshape(-1,1), delimiter=",")

np.savetxt("t1.csv",theta[0],delimiter=",")
np.savetxt("t2.csv",theta[1],delimiter=",")
np.savetxt("t3.csv", theta[2].reshape(1,1), delimiter=",")
