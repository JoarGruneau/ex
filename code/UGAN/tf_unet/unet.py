from __future__ import print_function, division, absolute_import, unicode_literals

import os
import shutil
import numpy as np
from collections import OrderedDict
import logging
import time

import tensorflow as tf

from tf_unet import util
from tf_unet.layers import (weight_variable, weight_variable_devonc, bias_variable, 
                            conv2d, deconv2d, max_pool, crop_and_concat, pixel_wise_softmax_2,
                            cross_entropy, batch_norm, conv2d_fixed_padding, block_layer)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

def create_conv_net(x, keep_prob, channels, n_class, unet_kwargs):
    """
    Creates a new convolutional unet for the given parametrization.
    
    :param x: input tensor, shape [?,nx,ny,channels]
    :param keep_prob: dropout probability tensor
    :param channels: number of channels in the input image
    :param n_class: number of output labels
    :param layers: number of layers in the net
    :param features_root: number of features in the first layer
    :param filter_size: size of the convolution filter
    :param pool_size: size of the max pooling operation
    """
    layers = unet_kwargs.pop('layers', 3)
    features_root = unet_kwargs.pop('features_root', 16)
    filter_size = unet_kwargs.pop('filter_size', 3)
    pool_size = unet_kwargs.pop('pool_size', 2)
    
    logging.info("Layers {layers}, features {features}, filter size {filter_size}x{filter_size}, pool size: {pool_size}x{pool_size}".format(layers=layers,
                                                                                                           features=features_root,
                                                                                                           filter_size=filter_size,
                                                                                                           pool_size=pool_size))
    # Placeholder for the input image
    nx = tf.shape(x)[1]
    ny = tf.shape(x)[2]
    x_image = tf.reshape(x, tf.stack([-1,nx,ny,channels]))
    in_node = x_image
    batch_size = tf.shape(x_image)[0]

    with tf.variable_scope('generator'):
        pools = OrderedDict()
        deconv = OrderedDict()
        dw_h_convs = OrderedDict()
        up_h_convs = OrderedDict()
        
        in_size = 1000
        size = in_size
        # down layers
        for layer in range(0, layers):
            features = 2**layer*features_root
            stddev = np.sqrt(2 / (filter_size**2 * features))
            if layer == 0:
                w1 = weight_variable([filter_size, filter_size, channels, features], stddev)
            else:
                w1 = weight_variable([filter_size, filter_size, features//2, features], stddev)
                
            w2 = weight_variable([filter_size, filter_size, features, features], stddev)
            b1 = bias_variable([features])
            b2 = bias_variable([features])
            
            conv1 = conv2d(in_node, w1, keep_prob)
            tmp_h_conv = tf.nn.relu(conv1 + b1)
            conv2 = conv2d(tmp_h_conv, w2, 1)
            dw_h_convs[layer] = tf.nn.relu(conv2 + b2)

            size -= 4
            if layer < layers-1:
                pools[layer] = max_pool(dw_h_convs[layer], pool_size)
                in_node = pools[layer]
                size /= 2

        in_node = dw_h_convs[layers-1]

        # up layers
        for layer in range(layers-2, -1, -1):
            features = 2**(layer+1)*features_root
            stddev = np.sqrt(2 / (filter_size**2 * features))

            wd = weight_variable_devonc([pool_size, pool_size, features//2, features], stddev)
            bd = bias_variable([features//2])
            h_deconv = tf.nn.relu(deconv2d(in_node, wd, pool_size) + bd)
            h_deconv_concat = crop_and_concat(dw_h_convs[layer], h_deconv)
            deconv[layer] = h_deconv_concat

            w1 = weight_variable([filter_size, filter_size, features, features//2], stddev)
            w2 = weight_variable([filter_size, filter_size, features//2, features//2], stddev)
            b1 = bias_variable([features//2])
            b2 = bias_variable([features//2])

            conv1 = conv2d(h_deconv_concat, w1, 1)
            h_conv = tf.nn.relu(conv1 + b1)
            conv2 = conv2d(h_conv, w2, 1)
            in_node = tf.nn.relu(conv2 + b2)
            up_h_convs[layer] = in_node

            size *= 2
            size -= 4

        # Output Map
        weight = weight_variable([1, 1, features_root, n_class], stddev)
        bias = bias_variable([n_class])
        conv = conv2d(in_node, weight, tf.constant(1.0))
        output_map = tf.nn.relu(conv + bias)
        up_h_convs["out"] = output_map

    return output_map, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'generator'), int(in_size - size)



class Ugan(object):
    """
    A unet implementation

    :param channels: (optional) number of channels in the input image
    :param n_class: (optional) number of output labels
    :param cost: (optional) name of the cost function. Default is 'cross_entropy'
    :param cost_kwargs: (optional) kwargs passed to the cost function. See Unet._get_cost for more options
    """

    def __init__(self, channels=3, n_class=2, cost="cross_entropy", adversarial=True,
                 border_addition=0, patch_size=1000, summaries=True, cost_kwargs={}, unet_kwargs={}):
        tf.reset_default_graph()

        self.n_class = n_class
        self.summaries = summaries

        self.x = tf.placeholder("float", shape=[None, None, None, channels])
        self.y = tf.placeholder("float", shape=[None, None, None, n_class])
        self.w = tf.placeholder("float", shape=[None, None, None])
        self.keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
        self.is_training = tf.placeholder(tf.bool)

        generator_logits, self.generator_variables, self.offset = create_conv_net(
            self.x, self.keep_prob, channels, n_class, unet_kwargs)

        self.border_addition = border_addition
        if border_addition != 0:
            generator_logits = generator_logits[:, border_addition:-border_addition,border_addition:-border_addition, ...]

        self.predicter = pixel_wise_softmax_2(generator_logits)
        self.bce_loss, self.pred = self._get_cost(generator_logits, cost, cost_kwargs)

        # self.cross_entropy = tf.reduce_mean(cross_entropy(tf.reshape(self.y, [-1, n_class]),
        #                                                   tf.reshape(pixel_wise_softmax_2(generator_logits),
        #                                                              [-1, n_class])))

        self.argmax = tf.argmax(self.predicter, 3)
        self.correct_pred = tf.equal(self.argmax, tf.argmax(self.y, 3))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        self.tp = tf.reduce_sum(tf.cast(tf.argmax(self.predicter, 3), tf.float32) * self.y[..., 1])
        self.fp = tf.reduce_sum(tf.cast(tf.argmax(self.predicter, 3), tf.float32)) - self.tp
        self.fn = tf.reduce_sum(self.y[..., 1]) - self.tp

        self.precision = self.tp / (self.tp + self.fp)
        self.recall = self.tp / (self.tp + self.fn)
        self.f1 = 2 * self.recall * self.precision / (self.recall + self.precision)


        # smooth_labels = smooth(self.y, 2, 0.1)*np.random.normal(0.95, 0.5)
        # print(smooth_labels.shape)
        # smooth_labels = tf.reshape(self.y[:,:,:,1]*np.random.normal(0.85, 0.15), (1, patch_size, patch_size, 1))
        # smooth_labels[...,0] = 1.0 - smooth_labels[...,1]
        # smooth_labels = tf.reshape(smooth_labels, (1, patch_size, patch_size, n_class))
        # smooth_labels = tf.concat([1.0 -smooth_labels, smooth_labels], axis=3)
        #prediction = tf.cast(tf.stack([1 - self.argmax, self.argmax], axis=3), tf.float32)
        # image_patches = tf.extract_image_patches(
        #     image, PATCH_SIZE, PATCH_SIZE, [1, 1, 1, 1], 'VALID')

        self.generator_cost=self.bce_loss


    def _get_cost(self, logits, cost_name, cost_kwargs):
        """
        Constructs the cost function, either cross_entropy, weighted cross_entropy or dice_coefficient.
        Optional arguments are:
        class_weights: weights for the different classes in case of multi-class imbalance
        regularizer: power of the L2 regularizers added to the loss function
        """

        flat_logits = tf.reshape(logits, [-1, self.n_class])
        flat_labels = tf.reshape(self.y, [-1, self.n_class])
        # flat_weights = tf.reshape(self.w,  [-1, 1])
        if cost_name == "cross_entropy":
            class_weights = cost_kwargs.pop("class_weights", None)

            if class_weights is not None:
                class_weights = tf.constant(np.array(class_weights, dtype=np.float32))

                weight_map = tf.multiply(flat_labels, class_weights)
                weight_map = tf.reduce_sum(weight_map, axis=1)

                loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,
                                                                   labels=flat_labels)
                weighted_loss = tf.multiply(loss_map, weight_map)

                loss = tf.reduce_mean(weighted_loss)

            else:
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(self.y[...,1], tf.int32), logits=logits)
                # class_weight = tf.div(tf.reduce_sum(self.y[..., 0]),tf.reduce_sum(self.y[..., 1]))
                weights=self.y[..., 0] + 2*self.y[..., 1]+10*self.w
                # loss = tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,
                #                                                               labels=flat_labels)
                loss = tf.reduce_mean(tf.multiply(loss, weights))
                # loss = tf.reduce_mean(loss)

                # loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,
                #                                                                labels=flat_labels)
                # weight = (1000000.0-tf.reduce_sum(loss_map[..., 1]))/tf.reduce_sum(loss_map[..., 1])
                # loss = (tf.reduce_sum(loss_map[..., 0])+tf.reduce_sum(loss_map[..., 1]*weight))/1000000.0
        # elif cost_name == "sigmoid_cross_entropy"

        elif cost_name == "dice_coefficient":
            eps = 1e-5
            prediction = pixel_wise_softmax_2(logits)
            intersection = tf.reduce_sum(prediction * self.y)
            union =  eps + tf.reduce_sum(prediction) + tf.reduce_sum(self.y)
            loss = -(2 * intersection/ (union))

        elif cost_name=='IoU':
            eps = 1e-5
            logits = pixel_wise_softmax_2(logits)
            inter_ground = tf.reduce_sum(logits[...,0] * self.y[...,0])
            inter_pred = tf.reduce_sum(logits[..., 1] * self.y[..., 1])
            ground_loss = -tf.div(inter_ground,
                                tf.reduce_sum(logits[...,0])
                                  + tf.reduce_sum(self.y[...,0]) -inter_ground +eps)
            pred_loss = -tf.div(inter_pred,
                                tf.reduce_sum(logits[...,1])
                                + tf.reduce_sum(self.y[...,1]) - inter_pred + eps)
            loss=1+pred_loss
            # loss = tf.cond(sum_labels_map > 0, lambda: 1.0 - tf.div(inter,union + eps), lambda: 0.0)


        else:
            raise ValueError("Unknown cost function: "%cost_name)

        regularizer = cost_kwargs.pop("regularizer", None)
        if regularizer is not None:
            regularizers = sum([tf.nn.l2_loss(variable) for variable in self.generator_variables])
            loss += (regularizer * regularizers)

        return loss, logits

    def predict(self, model_path, test_data_provider, test_iters, border_size, patch_size, input_size, name,
                prediction_path, verification_batch_size=1, combine=False, hard_prediction=True,
                filter_size=15, overlay=True, evaluate_scores = True):
        """
        Uses the model to create a prediction for the given data

        :param model_path: path to the model checkpoint to restore
        :param x_test: Data to predict on. Shape [n, nx, ny, channels]
        :returns prediction: The unet prediction Shape [n, px, py, labels] (px=nx-self.offset/2)
        """
        if not os.path.exists(prediction_path):
            logging.info("Allocating '{:}'".format(prediction_path))
            os.makedirs(prediction_path)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            # Initialize variables
            sess.run(init)

            # Restore model weights from previously saved model
            self.restore(sess, model_path)
            self.store_prediction(sess, test_iters, test_data_provider, border_size, patch_size, input_size, name,
                             prediction_path, verification_batch_size, combine=combine, hard_prediction=hard_prediction,
                                  filter_size=filter_size, overlay=overlay, evaluate_scores=evaluate_scores)

    def save(self, sess, model_path, global_step):
        """
        Saves the current session to a checkpoint

        :param sess: current session
        :param model_path: path to file system location
        """

        saver = tf.train.Saver()
        save_path = saver.save(sess, model_path, global_step=global_step)
        return save_path

    def restore(self, sess, model_path):
        """
        Restores a session from a checkpoint

        :param sess: current session instance
        :param model_path: path to file system checkpoint location
        """

        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        logging.info("Model restored from file: %s" % model_path)

    def store_prediction(self, sess, eval_iters, eval_data_provider, border_size, patch_size, input_size, name,
                         prediction_path, verification_batch_size, combine=False, hard_prediction=True,
                         logg_time=True, overlay = True, filter_size = 5, evaluate_scores=False):
        if evaluate_scores:
            scores = [[] for _ in range(6)]
        for i in range(eval_iters):
            patches = eval_data_provider.get_patches(get_coordinates=True)
            if combine or overlay or evaluate_scores:
                label = np.zeros((input_size, input_size, 2))
                if combine:
                    image = np.zeros((input_size, input_size, 3))

            prediction = np.zeros((input_size, input_size, self.n_class))
            for patch in patches:
                if logg_time:
                    start_time = time.time()
                pred = sess.run((self.predicter), feed_dict={self.x: patch[0],
                                                                 self.y: patch[1],
                                                                 self.keep_prob: 1.0,
                                                                 self.is_training: False})
                if logg_time:
                    duration = time.time() - start_time
                    logging.info("time: " + str(duration))
                x, y = patch[3]
                prediction[x:x + patch_size, y:y + patch_size, ...] = pred

                if combine or overlay or evaluate_scores:
                    label[x:x + patch_size, y:y + patch_size, ...] = patch[1]
                    if combine:
                        image[x:x + patch_size, y:y + patch_size, ...] = \
                            patch[0][0, border_size:-border_size, border_size:-border_size, ...]

            pred_shape = prediction.shape
            if combine:
                img = util.combine_img_prediction(image, label, prediction)
            else:
                if hard_prediction:
                    img = np.argmax(prediction, axis=2)
                    if overlay:
                        img = util.combine(util.filter_image(label[..., 1], filter_size),
                                                  util.filter_image(img, filter_size))
                else:
                    img = prediction[..., 1]
                img = util.to_rgb(img)
            util.save_image(img, "%s/%s_%s.jpg" % (prediction_path, name, i))

            if evaluate_scores:
                label = util.filter_image(label[..., 1], filter_size)
                prediction = util.filter_image(np.argmax(prediction, axis=2), filter_size)
                tmp_scores = util.calculate_f1_score(label, prediction)
                print(tmp_scores)
                for i in range(len(scores)):
                    scores[i].append(tmp_scores[i])
        print([np.mean(score) for score in scores])

        return pred_shape

    def calc_object_f1_scores(self, sess, eval_iters, eval_data_provider, border_size, patch_size, input_size, filter_size = 10):
        tp, fp, fn = 0.0
        for i in range(eval_iters):
            patches = eval_data_provider.get_patches(get_coordinates=True)
            label = np.zeros((input_size, input_size, 2))
            prediction = np.zeros((input_size, input_size, self.n_class))
            for patch in patches:
                pred = sess.run((self.predicter), feed_dict={self.x: patch[0],
                                                                 self.y: patch[1],
                                                                 self.keep_prob: 1.0,
                                                                 self.is_training: False})
                x, y = patch[3]
                prediction[x:x + patch_size, y:y + patch_size, ...] = pred
                label[x:x + patch_size, y:y + patch_size, ...] = patch[1]
            label = label[..., 1]
            prediction = util.filter_image(np.argmax(prediction, axis=2), filter_size)
            tmp_scores = util.calculate_f1_score(label, prediction)
            tp += tmp_scores[1]
            fp += tmp_scores[2]
            fn += tmp_scores[3]

        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        f1_score =2*precision*recall/(recall+precision)
        return precision, recall, f1_score





class Trainer(object):
    """
    Trains a unet instance

    :param net: the unet instance to train
    :param batch_size: size of training batch
    :param norm_grads: (optional) true if normalized gradients should be added to the summaries
    :param optimizer: (optional) name of the optimizer to use (momentum or adam)
    :param opt_kwargs: (optional) kwargs passed to the learning rate (momentum opt) and to the optimizer

    """

    verification_batch_size = 1

    def __init__(self, net, batch_size=1, norm_grads=False, optimizer="momentum", d_opt_kwargs={}, g_opt_kwargs={}):
        self.net = net
        self.batch_size = batch_size
        self.norm_grads = norm_grads
        self.optimizer = optimizer
        self.d_opt_kwargs = d_opt_kwargs
        self.g_opt_kwargs = g_opt_kwargs

    def _get_optimizer(self, training_iters, global_step):
        if self.optimizer == "momentum":
            learning_rate = self.opt_kwargs.pop("learning_rate", 0.2)
            decay_rate = self.opt_kwargs.pop("decay_rate", 0.95)
            momentum = self.opt_kwargs.pop("momentum", 0.2)

            self.learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate,
                                                        global_step=global_step,
                                                        decay_steps=training_iters,
                                                        decay_rate=decay_rate,
                                                        staircase=True)

            optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate_node, momentum=momentum,
                                                   **self.opt_kwargs).minimize(self.net.cost,
                                                                                global_step=self.global_step)
        elif self.optimizer == "adam":
            # learning_rate = self.opt_kwargs.pop("learning_rate", 0.001)
            # self.learning_rate_node = tf.Variable(learning_rate)

            g_optimizer = tf.train.AdamOptimizer(**self.g_opt_kwargs).minimize(self.net.generator_cost,
                                                                           global_step=self.global_step,
                                                                           var_list=self.net.generator_variables)

        return g_optimizer

    def _initialize(self, training_iters, output_path, restore, prediction_path):
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.g_optimizer= self._get_optimizer(training_iters, self.global_step)

        init = tf.global_variables_initializer()

        self.prediction_path = prediction_path
        abs_prediction_path = os.path.abspath(self.prediction_path)
        output_path = os.path.abspath(output_path)

        if not restore:
            logging.info("Removing '{:}'".format(abs_prediction_path))
            shutil.rmtree(abs_prediction_path, ignore_errors=True)
            logging.info("Removing '{:}'".format(output_path))
            shutil.rmtree(output_path, ignore_errors=True)

        if not os.path.exists(abs_prediction_path):
            logging.info("Allocating '{:}'".format(abs_prediction_path))
            os.makedirs(abs_prediction_path)

        if not os.path.exists(output_path):
            logging.info("Allocating '{:}'".format(output_path))
            os.makedirs(output_path)
        # print ("global step =" +str(global_step))
        return init

    def get_eval_variables(self, tags):
        return [self.net.__getattribute__(tag) for tag in tags]


    def eval_net(self, sess, feed_dict, optimizers=[], eval_metrics=[], eval_results=[]):
        results = sess.run(optimizers + eval_metrics, feed_dict=feed_dict)
        for i in range(len(optimizers),len(results)):
            eval_results[i-len(optimizers)].append(results[i])

    def eval_epoch(self, sess, data_provider, iters, optimizers, tags, feed_dict):
        metrics = self.get_eval_variables(tags)
        results = [[] for _ in range(len(metrics))]
        for _ in range(iters):
            patches = data_provider.get_patches()
            for patch in patches:
                feed_dict[self.net.x] = patch[0]
                feed_dict[self.net.y] = patch[1]
                feed_dict[self.net.w] = patch[2]
                self.eval_net(sess, feed_dict, optimizers=optimizers, eval_metrics=metrics, eval_results=results)
        return [np.mean(result) for result in results]


    def train(self, data_provider, eval_data_provider, output_path,
              training_iters=10, eval_iters=4, epochs=100, dropout=0.75, display_step=1,
              predict_step=50, restore=False, write_graph=False, prediction_path = 'prediction'):
        """
        Lauches the training process

        :param data_provider: callable returning training and verification data
        :param output_path: path where to store checkpoints
        :param training_iters: number of training mini batch iteration
        :param epochs: number of epochs
        :param dropout: dropout probability
        :param display_step: number of steps till outputting stats
        :param restore: Flag if previous model should be restored
        :param write_graph: Flag if the computation graph should be written as protobuf file to the output path
        :param prediction_path: path where to save predictions on each epoch
        """
        save_path = os.path.join(os.path.join(output_path, 'model'), "model.cpkt")
        if epochs == 0:
            return save_path

        init = self._initialize(training_iters, output_path, restore, prediction_path)

        with tf.Session() as sess:
            if write_graph:
                tf.train.write_graph(sess.graph_def, output_path, "graph.pb", False)

            sess.run(init)

            if restore:
                ckpt = tf.train.get_checkpoint_state(os.path.join(output_path, 'model'))
                if ckpt and ckpt.model_checkpoint_path:
                    self.net.restore(sess, ckpt.model_checkpoint_path)
                    print ("restored")

            border_size = data_provider.get_border_size()
            patch_size = data_provider.get_patch_size()
            input_size = data_provider.get_input_size()
            patch_len=input_size//patch_size


            summary_writer = tf.summary.FileWriter(os.path.join(output_path, 'train'), graph=sess.graph)
            eval_summary_writer=tf.summary.FileWriter(os.path.join(output_path, 'eval'))
            logging.info("Start optimization")
            curr_step=tf.train.global_step(sess, self.global_step)
            curr_epoch=curr_step//(training_iters*patch_len)

            epoch_tags = ['generator_cost']
            eval_tags = ['accuracy', 'precision', 'recall', 'f1']
            display_tags = epoch_tags + eval_tags
            feed_dict = {self.net.x: None, self.net.y: None, self.net.keep_prob: dropout, self.net.is_training: True}

            self.net.store_prediction(sess, eval_iters, eval_data_provider, border_size,
                                      patch_size, input_size, "epoch_%s" % 'init', self.prediction_path,
                                      self.verification_batch_size, combine=False, hard_prediction=True, filter_size=15,
                                      overlay=True, evaluate_scores=True)

            for epoch in range(curr_epoch,epochs):
                results = self.eval_epoch(sess, data_provider, training_iters, [self.g_optimizer],
                                epoch_tags if epoch % display_step != 0 else display_tags, feed_dict)

                if epoch % display_step == 0:
                    save_path = self.net.save(sess, save_path, self.global_step)
                    self.write_summary(summary_writer, epoch, display_tags, results)
                    self.write_logg(['epoch', 'type']+display_tags, [epoch, 'train'] + results)
                    self.output_minibatch_stats(sess, eval_summary_writer, eval_iters, epoch,
                                                eval_data_provider, eval_tags, 'eval')
                else:
                    self.write_logg(['epoch']+epoch_tags, [epoch]+results)
                    self.write_summary(summary_writer, epoch,epoch_tags, results)

                if epoch%predict_step == 0:
                    self.net.store_prediction(sess, eval_iters, eval_data_provider,  border_size,
                                          patch_size, input_size, "epoch_%s"%epoch, self.prediction_path,
                                          self.verification_batch_size, combine=True)



            logging.info("Optimization Finished!")
            save_path = self.net.save(sess, save_path, self.global_step)
            self.net.store_prediction(sess, eval_iters, eval_data_provider,  border_size, patch_size,
                                  input_size, "epoch_%s"%epochs, self.prediction_path,
                                      self.verification_batch_size, combine=True)
            return save_path

    def write_logg(self, tags, results):
        logg_string = ''
        for i in range(len(tags)):
            if type(results[i]) == np.float32:
                logg_string += ', {:}= {:.4f}'.format(tags[i], results[i])
            else:
                logg_string += ', {:}= {:}'.format(tags[i], results[i])
        logging.info(logg_string)
    
    def output_minibatch_stats(self, sess, summary_writer, eval_iters, step,
                               data_provider, tags, stats_type):
        feed_dict = {self.net.x: None, self.net.y: None, self.net.keep_prob: 1.0, self.net.is_training: False}
        results = self.eval_epoch(sess, data_provider, eval_iters, optimizers=[],
                                  tags=tags, feed_dict=feed_dict)
        self.write_summary(summary_writer, step, tags, results)
        self.write_logg(['epoch', 'type']+tags, [step, stats_type] + results)

    def write_summary(self, summary_writer, step, tags, results):
        summary = tf.Summary()
        for i in range(len(tags)):
            summary.value.add(tag=tags[i], simple_value=results[i])
        summary_writer.add_summary(summary, step)
        summary_writer.flush()
