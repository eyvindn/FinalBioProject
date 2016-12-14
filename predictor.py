"""
The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import datetime

import numpy as np
import tensorflow as tf

import reader

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "test",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None, "data_path")

FLAGS = flags.FLAGS


class TSSModel(object):
    """The PTB model."""

    def __init__(self, is_training, config):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps #This represents how far back it looks. In our case, a large window? IDK
        size = 2 #config.hidden_size #This is a vector in the original, but in our case it's ON or OFF, so just an int of size 1?

        # One hot encoding of nucleotides
        self._input_data = tf.placeholder(tf.float32, [batch_size, num_steps, config.num_classes])

        # One hot encoding of whether it is a TSS or not!
        self._targets = tf.placeholder(tf.int32, [batch_size, num_steps]) #This is less in rank becuase the VALUE IS THE CLASS INDEX BRO

        # Slightly better results can be obtained with forget gate biases
        # initialized to 1 but the hyperparameters of the model would need to be
        # different than reported in the paper.
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0)
        if is_training and config.keep_prob < 1:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                lstm_cell, output_keep_prob=config.keep_prob)

        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)
        self._initial_state = cell.zero_state(batch_size, tf.float32)

        # with tf.device("/cpu:0"):
        #   embedding = tf.get_variable("embedding", [vocab_size, size])
        #   inputs = tf.nn.embedding_lookup(embedding, self._input_data)

        inputs = self._input_data


        # if is_training and config.keep_prob < 1:
        #     inputs = tf.nn.dropout(inputs, config.keep_prob)

        # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
        # This builds an unrolled LSTM for tutorial purposes only.
        # In general, use the rnn() or state_saving_rnn() from rnn.py.
        #
        # The alternative version of the code below is:
        #
        # from tensorflow.models.rnn import rnn
        # inputs = [tf.squeeze(input_, [1])
        #           for input_ in tf.split(1, num_steps, inputs)]
        # outputs, state = rnn.rnn(cell, inputs, initial_state=self._initial_state)
        outputs = []
        state = self._initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)

        # Weight our super inbalanced class:

        ratio = 1.0 / config.num_steps
        class_weight = tf.constant([ratio, 1.0 - ratio])
        weighted_outputs = tf.mul(outputs, class_weight)  # shape [batch_size, 2]

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(weighted_outputs, self._targets))

        self._cost = cost = loss
        self._final_state = state
        self._final_output = tf.nn.softmax(weighted_outputs)

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op


class LargeConfig(object):
    """Large config."""
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 50
    max_epoch = 6
    max_max_epoch = 6
    keep_prob = 1.0 #0.5
    lr_decay = 0.5 #1 / 1.15
    batch_size = 500

    # ATGC
    num_classes = 4


class TestConfig(object):
    """Tiny config, for testing."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 10.0
    num_layers = 2
    num_steps = 50
    max_epoch = 10
    max_max_epoch = 10
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 40

    #ATGC
    num_classes = 4


def run_epoch(session, m, data, eval_op, writer, merged, windowsize, verbose=False):
    """Runs the model on the given data."""
    epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(m.initial_state)
    for step, (x, y) in enumerate(reader.promoter_iterator(data, m.batch_size,
                                                      m.num_steps)):

        cost, state, summary, final_output, _ = session.run([m.cost, m.final_state, merged, m._final_output, eval_op],
                                     {m.input_data: x,
                                      m.targets: y,
                                      m.initial_state: state})
        costs += cost
        iters += m.num_steps
        print(cost)
        if step % windowsize == 0:
            print(final_output)
            writer.add_summary(summary, step)

    return costs


def get_config():
    if FLAGS.model == "large":
        return LargeConfig()
    elif FLAGS.model == "test":
        return TestConfig()
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)


def main(_):
    print("WTF")

    raw_data = reader.load_promoter_data()
    train_data = raw_data
    config = get_config()
    eval_config = get_config()
    eval_config.batch_size = 1
    eval_config.num_steps = 1

    print("here")


    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = TSSModel(is_training=True, config=config)

        # with tf.variable_scope("model", reuse=True, initializer=initializer):
        #     mvalid = TSSModel(is_training=False, config=config)
        #     mtest = TSSModel(is_training=False, config=eval_config)

        saver = tf.train.Saver(tf.trainable_variables())

        tf.scalar_summary('loss', m._cost)

        merged = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter("logs/" + str(datetime.datetime.now()), session.graph)

        tf.initialize_all_variables().run()

        wpss = []
        for i in range(config.max_max_epoch):
            lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
            m.assign_lr(session, config.learning_rate * lr_decay)

            print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
            loss = run_epoch(session, m, train_data, m.train_op, writer, merged, config.num_steps, verbose=True)

            save_path = saver.save(session, "/tmp/model.ckpt")
            print("Model saved in file: %s" % save_path)
            # wpss.append(wps)
            #
            # print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
            # valid_perplexity, _ = run_epoch(session, mvalid, valid_data, tf.no_op())
            # print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

        # test_perplexity, _ = run_epoch(session, mtest, test_data, tf.no_op())
        # print("Test Perplexity: %.3f" % test_perplexity)
        # print("Mean wps: ", np.mean(wpss))
        # print("Std wps:", np.std(wpss))

if __name__ == "__main__":
    tf.app.run()