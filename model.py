import tensorflow as tf
import numpy as np

class VQVAE(object):
  def __init__(self, args, sess, name="vqvae"):
    self.input_size = args.input_size
    self.output_width = args.output_width
    self.input_channel = args.input_channel
    self.K, self.D = args.K, args.D
    self.beta = beta
    self.sess = sess
    self.max_grad_norm = args.max_grad_norm
    self.learning_rate = tf.Variable(float(args.learning_rate), trainable=False, name="learning_rate")
    self.lr_decay_op = self.learning_rate.assign(tf.multiply(self.learning_rate, args.lr_decay))
    self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
    self.global_step = tf.Variable(0, trainable=False)

    with tf.name_scope("data"):
      self.x_images = tf.placeholder(tf.float32, [None, self.input_dim])
      self.z = tf.placeholder(tf.float32, [None, self.latent_dim])

    with tf.name_scope("vqvae"):
      self.embed = tf.get_variable(shape=[self.K, self.D], initializer=initializer, name=name)
      with tf.variable_scope("encoder"):
        self.z_e = encoder(self.x_images)
      with tf.variable_scope("latent"):
        z_e_expand = tf.tile(tf.expand_dims(self.z_e, -2), [1,1,1,self.K,1])
        embed_expd = tf.reshape(self.embed, [1,1,1,self.K,self.D])
        k = tf.argmin(tf.norm(z_e_expand - embed_expd, axis=-1), -1)
        z_q = tf.gather(self.embed, k)
      with tf.variable_scope("decoder"):
        x_logits, self.x_recons = decoder(z_q)

    with tf.name_scope("loss"):
      self.rec_loss = tf.reduce_mean(tf.reduce_sum(
          tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logits, labels=self.x_images), 1))
      self.vq = tf.reduce_mean(tf.norm(tf.stop_gradient(self.z_e) - z_q, axis=-1)**2, axis=[0,1,2])
      self.commit = tf.reduce_mean(tf.norm(self.z_e - tf.stop_gradient(z_q), axis=-1)**2, axis=[0,1,2])
      self.loss = self.rec_loss + self.vq + self.beta * self.commit

      tf.summary.scalar("reconstruct_loss", self.rec_loss)
      tf.summary.scalar("vq_loss", self.vq)
      tf.summary.scalar("commit_loss", self.commit)

    with tf.name_scope('train'):
      encoder_vars = self.trainable_vars("encoder")
      decoder_vars = self.trainable_vars("decoder")
      for line in encoder_vars:
        print line 
      print "- " * 50
      for line in decoder_vars:
        print line       

      decoder_grads = self.optimizer.compute_gradients(self.rec_loss, var_list=decoder_vars)
      decoder_grads_and_vars = [(tf.clip_by_norm(g, self.max_grad_norm), v) for g, v in decoder_grads]
      latent_grads = self.optimizer.compute_gradients(self.rec_loss, var_list=z_q)
      # encoder_grads = self.optimizer.compute_gradients(self.z_e, var_list=encoder_vars, grad_loss=latent_grads) + 
      #     self.beta * self.optimizer.compute_gradients(self.commit, var_list=encoder_vars)
      encoder_grads = [(tf.gradients(self.z_e, var, latent_grads)[0] + self.beta * tf.gradients(self.commit, var)[0], 
          var) for var in encoder_vars]
      encoder_grads_and_vars = [(tf.clip_by_norm(g, self.max_grad_norm), v) for g, v in encoder_grads]
      embed_grads = self.optimizer.compute_gradients(self.vq, var_list=self.embed)

      self.train_vae_op = self.optimizer.apply_gradients(v_grads_and_vars, global_step=self.global_step)
      self.train_adv_op = self.optimizer.apply_gradients(a_grads_and_vars, global_step=self.global_step)

    init_op = tf.global_variables_initializer()
    self.sess.run(init_op)
    self.summary = tf.summary.merge_all()
    self.saver = tf.train.Saver(tf.global_variables())    

  def encoder(self, x_images):
    x = tf.reshape(x_images, [-1, self.output_width, self.output_width, self.input_channel])
    x = tf.layers.conv2d(x, 5, (4, 4), (2, 2), activation=tf.nn.relu, padding='same', name='en_layer1')
    x = tf.layers.conv2d(x, 10, (4, 4), (2, 2), activation=tf.nn.relu, padding='same', name='en_layer1')
    x = tf.layers.conv2d(x, 20, (4, 4), (2, 2), activation=tf.nn.relu, padding='same', name='en_layer1')
    return x

  def decoder(self, z):
    s4 = z.get_shape().as_list()[1]
    x = tf.nn.relu(self.deconv2d(z, 4, 5, 10, 2, [self.batch_size, s4, s4, 32], name='de_layer1'))
    x = tf.nn.relu(self.deconv2d(x, 4,10, 20, 2, [self.batch_size, s4*2, s4*2, 16], name='de_layer2'))
    x = self.deconv2d(x, 4, 20, 1, 2, [self.batch_size, s4*4, s4*4, self.input_channel], name='de_layer3')
    x_logits = tf.reshape(x, [-1, self.input_dim])
    x_recons = tf.nn.sigmoid(x)
    return x_logits, x_recons    

  def reconstruct(self, x_images):
    feed_dict = {self.x_images: x_images}
    return self.sess.run(self.reconstruct, feed_dict)

  def trainable_vars(self, scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

  def deconv2d(self, x, filter_size, n_in, n_out, strides, output_shape, name=None):
    w = self.weight_variable(shape=[filter_size, filter_size, n_out, n_in], name=name+'_w')
    b = self.bias_variable(shape=[n_out], name=name+'_b')
    output_shape = tf.stack(output_shape)
    x = tf.nn.conv2d_transpose(value=x, filter=w, output_shape=output_shape,
                               strides=[1, strides, strides, 1], padding='SAME', name=name)
    x = tf.nn.bias_add(x, b)
    return x

  def weight_variable(self, shape, name, initializer=None):
    initializer = tf.truncated_normal_initializer(stddev=0.1)
    if initializer:
      initializer = initializer
    return tf.get_variable(shape=shape, initializer=initializer, name=name)

  def bias_variable(self, shape, name, initializer=None):
    initializer = tf.constant_initializer(0.)
    if initializer:
      initializer = initializer
    return tf.get_variable(shape=shape, initializer=initializer, name=name)

  def generate(self, z, batch_size):
    feed_dict= {self.z: z, self.batch_size: batch_size}
    with tf.variable_scope("decoder", reuse=True):
      logits, x_recons = self.decoder(self.z)
    return self.sess.run(x_recons, feed_dict=feed_dict)

class AVB(object):
  def __init__(self, args, sess, name="AVB"):
    self.input_size = args.input_size
    self.latent_dim = args.latent_size
    self.output_width = args.output_width
    self.sess = sess
    self.max_grad_norm = args.max_grad_norm
    self.learning_rate = tf.Variable(float(args.learning_rate), trainable=False, name="learning_rate")
    self.learning_rate_decay_op = self.learning_rate.assign(
        tf.multiply(self.learning_rate, args.lr_decay))
    self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
    self.global_step = tf.Variable(0, trainable=False)

    with tf.name_scope("data"):
      self.x_images = tf.placeholder(tf.float32, [None, self.input_dim])
      self.z = tf.placeholder(tf.float32, [None, self.latent_dim])

    self.batch_size = tf.shape(self.x_images)[0]
    self.z_sampled = tf.random_normal([self.batch_size, self.latent_dim])

    with tf.name_scope("avb"):
      with tf.variable_scope("encoder"):
        z = self.encoder(self.x_images)
        tf.summary.histogram('sample_gaussian', z)
      with tf.variable_scope("decoder"):
        x_logits, x_recons = self.decoder(z)
      with tf.variable_scope("adversary"):
        d_real = self.adversary(z)
      with tf.variable_scope("adversary", reuse=True):
        d_fake = self.adversary(self.z_sampled)

    with tf.name_scope("loss"):
      rec_loss = 0.5 * tf.reduce_sum(
              tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logits, labels=self.x_images), 1)
      self.vae_loss = tf.reduce_mean(rec_loss + d_real)
      real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                 logits=d_real, labels=tf.ones_like(d_real)))
      fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                 logits=d_fake, labels=tf.zeros_like(d_fake)))
      self.adv_loss = real_loss + fake_loss
      self.rec_loss = tf.reduce_mean(rec_loss)

      tf.summary.scalar("reconstruct_loss", self.rec_loss)
      tf.summary.scalar("adv_loss", self.adv_loss)
      tf.summary.scalar("vae_loss", self.vae_loss)

    with tf.name_scope("loss_vars"):
      self.vae_loss_vars = self.trainable_vars("encoder") + self.trainable_vars("decoder")
      self.adv_loss_vars = self.trainable_vars("adversary")

    with tf.name_scope('train'):
      v_grads_and_vars = self.optimizer.compute_gradients(self.vae_loss, var_list=self.vae_loss_vars)
      v_grads_and_vars = [(tf.clip_by_norm(g, self.max_grad_norm), v) for g, v in v_grads_and_vars]
      a_grads_and_vars = self.optimizer.compute_gradients(self.adv_loss, var_list=self.adv_loss_vars)
      a_grads_and_vars = [(tf.clip_by_norm(g, self.max_grad_norm), v) for g, v in a_grads_and_vars]
      self.train_vae_op = self.optimizer.apply_gradients(v_grads_and_vars, global_step=self.global_step)
      self.train_adv_op = self.optimizer.apply_gradients(a_grads_and_vars, global_step=self.global_step)

    init_op = tf.global_variables_initializer()
    self.sess.run(init_op)
    self.summary = tf.summary.merge_all()
    self.saver = tf.train.Saver(tf.global_variables())

  def encoder(self, x_images):
    x = tf.reshape(x_images, [-1, 28, 28, 1])
    x = tf.layers.conv2d(x, 16, (5, 5), activation=tf.nn.elu, name='en_layer1')
    x = self.add_linear(x, name='eps_1')
    x = tf.layers.conv2d(x, 32, (5, 5), activation=tf.nn.elu, name='en_layer2')
    x = self.add_linear(x, name='eps_2')
    x = tf.layers.conv2d(x, 32, (5, 5), activation=tf.nn.elu, name='en_layer3')
    x = self.add_linear(x, name='eps_3')
    b, h, w, c = x.get_shape().as_list()
    x = tf.reshape(x, [-1, h * w * c])
    z = tf.layers.dense(x, self.latent_dim, activation=None, name='out_layer')
    return z

  def add_linear(self, x, name):
    eps = tf.random_normal(tf.stack([self.batch_size, self.latent_dim]))
    target_shape = x.get_shape().as_list()
    target_size = int(np.prod([int(s) for s in target_shape[1:]]))
    outputs = tf.layers.dense(eps, target_size, 
              kernel_initializer=tf.contrib.layers.xavier_initializer(), name=name)
    outputs = tf.reshape(outputs, tf.shape(x))
    outputs += x
    return outputs

  def decoder(self, z):
    s = float(self.output_width)
    s2, s4, s8 = int(np.ceil(s/2)), int(np.ceil(s/4)), int(np.ceil(s/8))
    x = tf.layers.dense(z, s8 * s8 * 32, activation=tf.nn.relu, name='fc_layer')
    x = tf.reshape(x, [-1, s8, s8, 32])

    x = tf.nn.relu(self.deconv2d(x, 5, 32, 32, 2, [self.batch_size, s4, s4, 32], name='de_layer1'))
    x = tf.nn.relu(self.deconv2d(x, 5, 32, 16, 2, [self.batch_size, s2, s2, 16], name='de_layer2'))
    x = self.deconv2d(x, 5, 16, 1, 2, [self.batch_size, 28, 28, 1], name='de_layer3')

    x_logits = tf.reshape(x, [-1, self.input_dim])
    x_recons = tf.nn.sigmoid(x)
    return x_logits, x_recons

  def adversary(self, inputs):
    x = tf.layers.dense(inputs, 256, activation=tf.nn.elu, name='adv_layer1')
    x = tf.layers.dense(x, 128, activation=tf.nn.elu, name='adv_layer2')
    score = tf.layers.dense(x, 1, activation=None, name='adv_layer3')
    return score

  def train_adv(self, x_images):
    feed_dict = {self.x_images: x_images}
    return self.sess.run([self.train_adv_op, self.adv_loss, self.global_step, self.summary], feed_dict=feed_dict)

  def train_vae(self, x_images):
    feed_dict = {self.x_images: x_images}
    return self.sess.run([self.train_vae_op, self.vae_loss, self.rec_loss, 
                          self.global_step, self.summary], feed_dict=feed_dict)

  def generate(self, z):
    feed_dict= {self.z: z}
    with tf.variable_scope("decoder", reuse=True):
      logits, x_recons = self.decoder(self.z)
    return self.sess.run(x_recons, feed_dict=feed_dict)

  def reconstruct(self, x_images):
    feed_dict = {self.x_images: x_images}
    return self.sess.run(self.reconstruct, feed_dict)

  def trainable_vars(self, scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

  def deconv2d(self, x, filter_size, n_in, n_out, strides, output_shape, name=None):
    w = self.weight_variable(shape=[filter_size, filter_size, n_out, n_in], name=name+'_w')
    b = self.bias_variable(shape=[n_out], name=name+'_b')
    output_shape = tf.stack(output_shape)
    x = tf.nn.conv2d_transpose(value=x, filter=w, output_shape=output_shape,
                               strides=[1, strides, strides, 1], padding='SAME', name=name)
    x = tf.nn.bias_add(x, b)
    return x

  def weight_variable(self, shape, name, initializer=None):
    initializer = tf.truncated_normal_initializer(stddev=0.1)
    if initializer:
      initializer = initializer
    return tf.get_variable(shape=shape, initializer=initializer, name=name)

  def bias_variable(self, shape, name, initializer=None):
    initializer = tf.constant_initializer(0.)
    if initializer:
      initializer = initializer
    return tf.get_variable(shape=shape, initializer=initializer, name=name)

  def generate(self, z, batch_size):
    feed_dict= {self.z: z, self.batch_size: batch_size}
    with tf.variable_scope("decoder", reuse=True):
      logits, x_recons = self.decoder(self.z)
    return self.sess.run(x_recons, feed_dict=feed_dict)