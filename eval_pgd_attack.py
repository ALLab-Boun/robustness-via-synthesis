"""
Modified from https://github.com/MadryLab/cifar10_challenge.git
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os

#os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import cifar10_input

def convert_to_onehot(y, nb_classes):
    y_onehot = np.zeros([len(y), nb_classes])
    for i in range(len(y)):
        y_onehot[i, y[i]] = 1.0
    return y_onehot



class LinfPGDAttack:
  def __init__(self, model, epsilon, num_steps, step_size, random_start, loss_func):
    """Attack parameter initialization. The attack performs k steps of
       size a, while always staying within epsilon from the initial
       point."""
    self.model = model
    self.epsilon = epsilon
    self.num_steps = num_steps
    self.step_size = step_size
    self.rand = random_start
    
    
    mean_xent, xent, decay, xent_adv, ot_loss, accuracy_adv, num_correct_nat, pre_softmax = self.model.loss_func()


    if loss_func == 'xent':
      loss = xent
    elif loss_func == 'cw':
      label_mask = model.y_input
      correct_logit = tf.reduce_sum(label_mask * pre_softmax, axis=1)
      wrong_logit = tf.reduce_max((1-label_mask) * pre_softmax - 1e4*label_mask, axis=1)
      loss = -tf.nn.relu(correct_logit - wrong_logit + 50)
    else:
      print('Unknown loss function. Defaulting to cross-entropy')
      

    self.grad = tf.gradients(loss, model.x_input)[0]

  def perturb(self, x_nat, y, sess):
    """Given a set of examples (x_nat, y), returns a set of adversarial
       examples within epsilon of x_nat in l_infinity norm."""
    if self.rand:
      rand_init = np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
      x = x_nat + rand_init 
      x = np.clip(x, 0, 255) # ensure valid pixel range
    else:
      x = x_nat.astype(np.float)

    for i in range(self.num_steps):
        
      grad = sess.run(self.grad, feed_dict={self.model.x_input: x, self.model.y_input: y})

      x = np.add(x, self.step_size * np.sign(grad), out=x, casting='unsafe')

      x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
      x = np.clip(x, 0, 255) # ensure valid pixel range

    return x


if __name__ == '__main__':
  import json
  import sys
  import math


  from model import Model

  with open('config.json') as config_file:
    config = json.load(config_file)

  model_file = tf.train.latest_checkpoint(config['model_dir'])
  print("Restored from %s", model_file)
  if model_file is None:
    print('No model found')
    sys.exit()

  data_path = config['data_path']

  model = Model(mode='eval', epsilon=config['epsilon'], num_class=10)

  mean_xent, xent, weight_decay_loss, xent_adv, ot_loss, accuracy_adv, num_correct_nat, _ = model.loss_func()

  attack = LinfPGDAttack(model,
                         config['epsilon'],
                         config['num_steps'],
                         config['step_size'],
                         config['random_start'],
                         config['loss_func'])
  saver = tf.train.Saver()

  cifar = cifar10_input.CIFAR10Data(data_path)

  with tf.Session() as sess:
    # Restore the checkpoint
    saver.restore(sess, model_file)

    # Iterate over the samples batch-by-batch
    num_eval_examples = config['num_eval_examples']
    eval_batch_size = config['eval_batch_size']
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

    x_adv = [] # adv accumulator

    print('Iterating over {} batches'.format(num_batches))

    total_corr = 0
    total_corr_nat = 0

    for ibatch in range(num_batches):
      bstart = ibatch * eval_batch_size
      bend = min(bstart + eval_batch_size, num_eval_examples)
      print('batch size: {}'.format(bend - bstart))

      x_batch = cifar.eval_data.xs[bstart:bend, :]
      y_batch = cifar.eval_data.ys[bstart:bend]
      y_batch = convert_to_onehot(y_batch, 10)

      x_batch_adv = attack.perturb(x_batch, y_batch, sess)
      dict_nat = {model.x_input: x_batch, model.y_input: y_batch}

      x_adv.append(x_batch_adv)

      dict_adv = {model.x_input: x_batch_adv, model.y_input: y_batch}

      cur_corr_adv = sess.run(num_correct_nat, feed_dict=dict_adv)
      cur_corr_nat = sess.run(num_correct_nat, feed_dict=dict_nat)

      total_corr += cur_corr_adv
      total_corr_nat += cur_corr_nat


    acc_adv = total_corr / num_eval_examples
    acc_nat_nat = total_corr_nat / num_eval_examples

    print('Accuracy: {:.2f}%'.format(100.0 * acc_adv))
    print('Natural Accuracy: {:.2f}%'.format(100.0 * acc_nat_nat))
