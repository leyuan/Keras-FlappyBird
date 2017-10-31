import sys
sys.path.append("game/")
import train
import numpy as np
import tensorflow as tf
import wrapped_flappy_bird as game
from skimage import transform, color, exposure

def load_model():
    model = train.build_model()
    model.load_weights('ly_model.h5')
    return model


def main():
    model = load_model()
    a_t = np.zeros(train.NUM_ACTIONS)
    a_t[0] = 1.

    game_state = game.GameState()
    s_t, r_t, terminal = game_state.frame_step(a_t)
    s_t = train.process_image(s_t)
    s_t = np.stack((s_t, s_t, s_t, s_t), axis=2)
    print(s_t.shape)
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])
    t = 0

    while(True):
        a_t1 = np.zeros(train.NUM_ACTIONS)

        q = model.predict(s_t)
        q_index = np.argmax(q)
        a_t1[q_index] = 1.

        s_t1, r_t1, terminal = game_state.frame_step(a_t1)
        s_t1 = train.process_image(s_t1)
        s_t1 = s_t1.reshape(1, s_t1.shape[0], s_t1.shape[1], 1)
        s_t1 = np.append(s_t1, s_t[:, :, :, :3], axis=3)

        s_t = s_t1
        t += 1

        print ("TIMESTAMP", t)

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)
    main()
