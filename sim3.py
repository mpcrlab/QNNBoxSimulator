import tensorflow as tf
import numpy as np
import random
from collections import deque
from RoverSim import Rover

ACTIONS = 3
GAMMA = 0.99
OBSERVE = 500
EXPLORE = 500
FINAL_EPSILON = 0.05
INITIAL_EPSILON = 1.0
REPLAY_MEMORY = 590000
BATCH = 32
K = 1

rover = Rover()

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def createNetwork():
    s = tf.placeholder(tf.float32, [None, 12])

    W = weight_variable([12, 6])
    b = bias_variable([6])

    W_out = weight_variable([6, ACTIONS])
    b_out = bias_variable([ACTIONS])

    y = tf.nn.softmax(tf.matmul(s, W) + b)
    readout = tf.matmul(y, W_out) + b_out

    return s, readout, y

def trainNetwork(s, readout, h_fc1, sess):
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.mul(readout, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)

    sess.run(tf.initialize_all_variables())

    D = deque()

    a_file = open("logs/readout.txt", "w")
    h_file = open("logs/hidden.txt", "w")

    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = rover.frame_step(do_nothing)
    s_t = x_t

    epsilon = INITIAL_EPSILON
    t = 0
    while t < 2000:
        readout_t = readout.eval(feed_dict = {s : [s_t]})[0]
        a_t = np.zeros([ACTIONS])
        action_index = 0
        if random.random() <= epsilon or t <= OBSERVE:
            action_index = random.randrange(ACTIONS)
            a_t[action_index] = 1
        else:
            action_index = np.argmax(readout_t)
            a_t[action_index] = 1

        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        for i in xrange(0, K):
            x_t1, r_t, terminal = rover.frame_step(a_t)
            s_t1 = x_t1

            D.append((s_t, a_t, r_t, s_t1, terminal))
            if len(D) > REPLAY_MEMORY:
                D.popleft()

        if t > OBSERVE:
            minibatch = random.sample(D, BATCH)

            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []
            readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch})
            for i in xrange(0, len(minibatch)):
                if minibatch[i][4]:
                    y_batch.append(r_batch[i])
                    #print "1: ", r_batch[i]
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))
                    #print "2: ", GAMMA * np.max(readout_j1_batch[i])

            train_step.run(feed_dict={y : y_batch, a : a_batch, s : s_j_batch})

        s_t = s_t1
        t += 1

        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"
        #print "TIMESTEP", t, "/ STATE", state, "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, "/ Q_MAX %e" % np.max(readout_t)

def test(s, readout, sess):
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    rover.state = rover.start
    x_t, r_0, terminal = rover.frame_step(do_nothing)
    s_t = x_t

    t = 0
    terminal = False
    while not terminal:
        readout_t = readout.eval(feed_dict = {s : [s_t]})[0]
        a_t = np.zeros([ACTIONS])
        action_index = 0
        action_index = np.argmax(readout_t)
        a_t[action_index] = 1

        for i in xrange(0, K):
            x_t1, r_t, terminal = rover.frame_step(a_t)
            s_t1 = x_t1

        s_t = s_t1
        t += 1
    return t, rover.minPath()

def run():
    sess =  tf.InteractiveSession()
    x, readout, y = createNetwork()
    trainNetwork(x, readout, y, sess)
    return test(x, readout, sess)

if __name__ == "__main__":
    print run()