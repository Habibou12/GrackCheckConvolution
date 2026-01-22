import time

import numpy as np
import cv2
import os


cat = []

new_size = [50,50]
threshold = 30
i = 0
for name in os.listdir("PetImages/Cat"):
    i += 1
    path = "PetImages/Cat/" + name
    img  = cv2.imread(path)
    if i > threshold:
        break
    if img is not None:
        img = cv2.resize(img, new_size)

        cat.append(img)

i = 0

dog = []
for name in os.listdir("PetImages/Dog"):
    i += 1
    path = "PetImages/Dog/" + name
    img  = cv2.imread(path)
    if i > threshold:
        break
    if img is not None:
        img = cv2.resize(img, new_size)
        dog.append(img)

cat = np.array(cat)
dog = np.array(dog)

y_dog = np.ones((dog.shape[0], 1))
y_cat = np.zeros((cat.shape[0], 1))

x_train = np.concatenate([dog, cat])
x_train = x_train/255.0
y_train = np.concatenate([y_dog, y_cat])


def initialiseParameters():
    parameters = {"cW1": np.random.randn(5,5,3, 2)*np.sqrt(1/200), "cb1":np.zeros((1,1,1,2)), "W2": np.random.randn(1058,16)*np.sqrt(1/1058), "b2":np.zeros((1,16)),
                  "W3": np.random.randn(16,1), "b3": np.zeros((1,1))}

    return parameters


def conv_single_step(a_slice_prev, W):


    s = np.multiply(a_slice_prev, W)
    Z = np.sum(s)
    Z = Z
    return Z


def sigmoid(z):
    return 1/(1 + np.exp(-z))

def Computecost(A3, Y):

    m = A3.shape[0]
    eps = 1e-15
    A2 = np.clip(A3, eps, 1- eps)
    loss = -np.sum(Y*np.log(A2) + (1- Y)*np.log(1- A2))
    return loss/m


def relu(z):
    return np.maximum(0,z)
def create_mask(x):
    mask = x == np.max(x)
    return mask

def forward(parameters, X, Y):
    cW1 = parameters["cW1"]
    W2 = parameters["W2"]
    cb1 = parameters["cb1"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    m, heigth, width, channel = X.shape
    f,f, channel, n_Cprime = cW1.shape

    n_H = int((heigth - f + 1))
    n_W = int((width- f + 1))

    Z1 = np.zeros((m, n_H, n_W, n_Cprime))

    for i in range(m):
        this_image = X[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_Cprime):
                    vert_start = h
                    vert_end = h  + f
                    horiz_start = w
                    horiz_end = w  + f
                    a_slice_prev = this_image[vert_start:vert_end, horiz_start:horiz_end]

                    weights = cW1[:, :, :, c]
                    result = conv_single_step(a_slice_prev, weights)
                    Z1[i, h, w, c] = result
    Z1 = Z1 + cb1

    A1 = Z1*np.where(Z1 > 0, 1, 0)

    stride = 2
    f = 2
    pool_N_H = int( 1 + (n_H - f)/stride)
    pool_N_W = int(1 + (n_W - f) / stride)

    APool = np.zeros((m,pool_N_H, pool_N_W, n_Cprime))

    for i in range(m):
        this_slice = A1[i]
        for h in range(pool_N_H):
            for w in range(pool_N_W):

                for c in range(n_Cprime):
                    this_conv = this_slice[h * stride:h * stride + f, w * stride:w * stride + f, c]
                    APool[i, h, w, c] = np.max(this_conv)


    APoolFlatten = APool.reshape(APool.shape[0],-1)

    Z2 = APoolFlatten.dot(W2) + b2
    A2 = relu(Z2)
    Z3 = A2.dot(W3) + b3
    A3 = sigmoid(Z3)

    cache = {"A3": A3, "A2": A2 ,"A1": A1, "Z1":Z1, "APool": APool,"ApoolFlatten": APoolFlatten}
    cost = Computecost(A3, Y)


    return cache, cost
def backward(parameters, X, Y):
    cache, _ = forward(parameters, X, Y)

    cW1 = parameters["cW1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]

    A3 = cache["A3"]
    A2 = cache["A2"]
    A1 = cache["A1"]
    Z1 = cache["Z1"]
    APoolFlatten = cache["ApoolFlatten"]
    APool = cache["APool"]

    batchSize= X.shape[0]


    dZ3 = (A3 - Y)/batchSize
    dW3 = A2.T.dot(dZ3)
    db3 = np.sum(dZ3, axis=0)

    dA2 = dZ3.dot(W3.T)
    dZ2 = dA2*np.where(A2 > 0, 1, 0)
    dW2 = (APoolFlatten.T.dot(dZ2))
    db2 = np.sum(dZ2, axis=0)

    dAPoolFlatten = dZ2.dot(W2.T)
    dAPool = np.reshape(dAPoolFlatten, APool.shape)
    m, n_H_prev, n_W_prev, n_C_prev = A1.shape
    m, n_H, n_W, n_C = dAPool.shape
    dA1 = np.zeros(( m, n_H_prev, n_W_prev, n_C_prev))
    stride = 2
    f = 2

    for i in range(m):
        this_aprev = A1[i]

        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride
                    vert_end = h * stride + f
                    horiz_start = w * stride
                    horiz_end = w * stride + f

                    a_prev_slice = this_aprev[vert_start:vert_end, horiz_start:horiz_end, c]

                    mask = create_mask(a_prev_slice)

                    dA1[i, vert_start: vert_end, horiz_start: horiz_end, c] += mask * dAPool[i, h, w, c]


    dZ1 = dA1*np.where(A1 > 0, 1, 0)


    (m, n_H, n_W, n_C) = dZ1.shape
    f, f, n_C_prev, n_C = cW1.shape

    dcW1 = np.zeros((f, f, n_C_prev, n_C))
    dcb1 = np.zeros((1,1,1, n_C))

    for i in range(m):
        this_image = X[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):

                    vert_start = h
                    vert_end = h  + 5
                    horiz_start = w
                    horiz_end = w + 5

                    a_slice = this_image[vert_start:vert_end, horiz_start:horiz_end]
                    dcW1[:,:,:,c] += a_slice * dZ1[i,h,w,c]
                    dcb1[:,:,:,c] += dZ1[i,h,w,c]
    dcW1 = dcW1

    grads = {"dcW1": dcW1, "dcb1":dcb1, "dW2": dW2, "db2": db2, 'dW3': dW3, "db3": db3}

    return grads

def dictionnary_to_vector(parameters):
    result = []
    for key in parameters:
        param = parameters[key].reshape(-1)
        result.extend(param)

    return np.array(result)

def vector_to_dictionnary(flatten_vector):
    parameters = {}
    start_idx = 0
    for key in shapes:
        shape = shapes[key]
        size = np.prod(shape)
        end_idx = start_idx + size
        parameters[key] = np.reshape(flatten_vector[start_idx:end_idx], shape)
        start_idx = end_idx


    return parameters



def gradscheck(parameters, X, Y):
    gradients = backward(parameters, X, Y)
    parameters_values = dictionnary_to_vector(parameters)
    num_param = parameters_values.shape[0]
    grads = dictionnary_to_vector(gradients)
    gradapprox = np.zeros(num_param)
    print(num_param)

    start_time = time.time()


    for i in range(num_param):
        theta_plus = np.copy(parameters_values)
        theta_plus[i] = theta_plus[i] + epsilon
        _,  J_plus = forward(vector_to_dictionnary(theta_plus), X, Y)


        theta_minus = np.copy(parameters_values)
        theta_minus[i] = theta_minus[i] - epsilon
        _, J_minus = forward(vector_to_dictionnary(theta_minus), X, Y)
        gradapprox[i] = (J_plus - J_minus)/(2*epsilon)


        temps_ecouler = (time.time() - start_time)
        iteration_actuelle = i + 1
        if i > 2:

           temps_par_iter = temps_ecouler/iteration_actuelle
           temps_total = temps_par_iter*num_param
           temps_restant =  temps_total - temps_ecouler
           print(str(temps_restant) + "s")


    numerator = np.linalg.norm(grads - gradapprox)
    denominator = np.linalg.norm(grads)  +  np.linalg.norm(gradapprox)
    difference = numerator/denominator
    print(difference)


x= np.random.rand(2,14,14,3)
y = np.ones((2,1))

parameters = initialiseParameters()

shapes = {key: parameters[key].shape for key in parameters}
epsilon = 1e-7


gradscheck(parameters, x_train[0:2], y[0:2])
