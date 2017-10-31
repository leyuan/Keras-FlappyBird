import sys
sys.path.append("game/")

import json
import random
import numpy as np
import tensorflow as tf
import wrapped_flappy_bird as game

import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer

from keras import initializers
from keras.initializers import normal, identity
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import SGD , Adam
from collections import deque

IMAGE_DIM = (84, 84)
INPUT_SHAPE = IMAGE_DIM + (4, )
LEARNING_RATE = 1e-4
REPLAY_MEMORY = 50000
OBSERVATION = 3200.
BATCH = 32
NUM_ACTIONS = 2
GAMMA = 0.99
EXPLORE = 3000000.

def process_image(s_t):
    s_t = color.rgb2gray(s_t)
    s_t = transform.resize(s_t, IMAGE_DIM)
    s_t = exposure.rescale_intensity(s_t, out_range=(0, 255))
    return s_t

def build_model():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=8, strides=(4, 4), data_format='channels_last', input_shape=INPUT_SHAPE))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=64, kernel_size=4, strides=(2, 2), data_format='channels_last'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), data_format='channels_last'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(2)) # only 2 actions to flap or not
    # model.add(Activation('linear'))

    adam=Adam(lr=LEARNING_RATE)
    model.compile(optimizer=adam, loss='mse')
    return model


def main():
    D = deque()
    initial_epsilon = 1
    final_epsilon = 0.1
    initial_action = [1, 0] # not flap at the beginning

    game_state = game.GameState()
    image_data, reward_0, terminal_0 = game_state.frame_step(initial_action)

    # initial state s_t, stack 4 together to get the wanted shape
    s_t = process_image(image_data)
    s_t = np.stack((s_t, s_t, s_t, s_t), axis=2)
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])
    # print (s_t.shape) # (1, 84, 84, 4) required by tensorflow

    timestamp = 0
    model = build_model()

    while (True):
        loss = 0
        action = np.zeros([NUM_ACTIONS])

        # Explore vs Exploit
        if random.random() <= initial_epsilon:
            print "----- RANDOM ACTION -----*"
            action_index = random.randrange(2)
            action[action_index] = 1
        else:
            q_values = model.predict(s_t)
            action_index = np.argmax(q_values)
            action[action_index] = 1

        if initial_epsilon > final_epsilon:
            initial_epsilon -= 0.0001

        image_data, reward_1, terminal_1 = game_state.frame_step(action)
        s_t1 = process_image(image_data)
        s_t1 = s_t1.reshape(1, s_t1.shape[0], s_t1.shape[1], 1)
        s_t1 = np.append(s_t1, s_t[:, :, :, :3], axis=3)

        # save transition
        D.append((s_t, action_index, reward_1, s_t1, terminal_1))

        if timestamp > OBSERVATION:
            loss = experience_replay(D, model, loss)

        timestamp += 1
        s_t = s_t1

        if timestamp % 1000 == 0:
            save_model(model)

        # clear D when necessary
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        print("TIMESTAMP", timestamp, "/ ACTION INDEX", action_index, "/ LOSS", round(loss, 2), \
        "/ EPSILON", round(initial_epsilon, 4), "/ REWARD", reward_1)


def experience_replay(replay_q, model, loss):
    minibatches = random.sample(replay_q, BATCH)

    inputs = np.zeros((BATCH,) + INPUT_SHAPE)
    targets = np.zeros((BATCH, NUM_ACTIONS))

    for i in range(len(minibatches)):
        s_0, action_index, reward, s_1, terminal = minibatches[i]

        inputs[i] = s_0
        targets[i] = model.predict(s_0)
        Q_sa = model.predict(s_1)

        # update reward
        if terminal:
            targets[i, action_index] = reward
        else:
            targets[i, action_index] = reward + GAMMA * np.max(Q_sa)

    loss += model.train_on_batch(inputs, targets)
    return loss

def save_model(model):
    print("Now we save model")
    model.save_weights("ly_model.h5", overwrite=True)
    with open("ly_model.json", "w") as outfile:
        json.dump(model.to_json(), outfile)

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)
    main()
