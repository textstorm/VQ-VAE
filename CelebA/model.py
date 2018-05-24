import tensorflow as tf
import numpy as np

class VQVAE(object):
  def __init__(self, args, sess, name="vqvae"):
    self.input_size = args.input_size
    self.output_width = args.output_width
    self.input_channel = args.input_channel
    self.K, self.D = args.K, args.D
    self.beta = args.beta
    self.sess = sess
    self.max_grad_norm = args.max_grad_norm
    self.learning_rate = tf.Variable(float(args.learning_rate), trainable=False, name="learning_rate")
    self.lr_decay_op = self.learning_rate.assign(tf.multiply(self.learning_rate, args.lr_decay))
    self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
    self.global_step = tf.Variable(0, trainable=False)

    with tf.name_scope("data"):
      self.x_images = tf.placeholder(tf.float32, [None] + self.input_size)

    self.batch_size = tf.shape(self.x_images)[0]
    with tf.name_scope("vqvae"):
      self.embed = tf.get_variable(shape=[self.K, self.D], 
          initializer=tf.truncated_normal_initializer(stddev=0.02), name=name)
      tf.summary.histogram("embedding", self.embed)
      with tf.variable_scope("encoder"):
        self.z_e = self.encoder(self.x_images)
      with tf.variable_scope("latent"):
        z_e_expand = tf.tile(tf.expand_dims(self.z_e, -2), [1,1,1,self.K,1])
        embed_expd = tf.reshape(self.embed, [1,1,1,self.K,self.D])
        k = tf.argmin(tf.norm(z_e_expand - embed_expd, axis=-1), -1)
        self.k = k
        z_q = tf.gather(self.embed, k)
      with tf.variable_scope("decoder"):
        x_logits, self.x_recons = self.decoder(z_q)

    with tf.name_scope("loss"):
      # self.rec_loss = tf.reduce_mean(tf.reduce_sum(
      #       tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logits, labels=self.x_images), axis=[1,2,3]))
      self.rec_loss = tf.reduce_mean(tf.reduce_sum((self.x_images - self.x_recons)**2, axis=[1, 2, 3]))
      self.vq = tf.reduce_mean(tf.norm(tf.stop_gradient(self.z_e) - z_q, axis=-1)**2, axis=[0,1,2])
      self.commit = tf.reduce_mean(tf.norm(self.z_e - tf.stop_gradient(z_q), axis=-1)**2, axis=[0,1,2])
      self.loss = self.rec_loss + self.vq + self.beta * self.commit

      tf.summary.scalar("reconstruct_loss", self.rec_loss)
      tf.summary.scalar("vq_loss", self.vq)
      tf.summary.scalar("commit_loss", self.commit)

    with tf.name_scope('train'):
      encoder_vars = self.trainable_vars("encoder")
      decoder_vars = self.trainable_vars("decoder")  

      decoder_grads = list(zip(tf.gradients(self.rec_loss, decoder_vars), decoder_vars))
      decoder_grads_and_vars = [(tf.clip_by_norm(g, self.max_grad_norm), v) for g, v in decoder_grads]
      latent_grads = tf.gradients(self.rec_loss, z_q)
      encoder_grads = [(tf.gradients(self.z_e, var, latent_grads)[0] + self.beta * tf.gradients(self.commit, var)[0], 
          var) for var in encoder_vars]
      encoder_grads_and_vars = [(tf.clip_by_norm(g, self.max_grad_norm), v) for g, v in encoder_grads]
      embed_grads = list(zip(tf.gradients(self.vq, self.embed), [self.embed]))
      embed_grads_and_vars = [(tf.clip_by_norm(g, self.max_grad_norm), v) for g, v in embed_grads]

      self.train_op = self.optimizer.apply_gradients(decoder_grads_and_vars + encoder_grads_and_vars
            + embed_grads_and_vars, global_step=self.global_step)

    init_op = tf.global_variables_initializer()
    self.sess.run(init_op)
    self.summary = tf.summary.merge_all()
    self.saver = tf.train.Saver(tf.global_variables())    

  def encoder(self, x_images):
    x = tf.layers.conv2d(x_images, self.D//2, (4, 4), (2, 2), activation=tf.nn.leaky_relu, padding='same', name='en_layer1')
    x = tf.layers.conv2d(x, self.D//2, (4, 4), (2, 2), activation=tf.nn.leaky_relu, padding='same', name='en_layer2')
    x = tf.layers.conv2d(x, self.D, (4, 4), (2, 2), activation=tf.nn.leaky_relu, padding='same', name='en_layer3')
    x = self.residual(x, "res_layer1")
    x = self.residual(x, "res_layer2")
    return x

  def decoder(self, z):
    s = float(self.output_width)
    s2, s4, s8 = int(np.ceil(s/2)), int(np.ceil(s/4)), int(np.ceil(s/8))
    x = self.residual(z, "res_layer1")
    x = self.residual(x, "res_layer2")
    x = tf.nn.relu(self.deconv2d(x, 4, self.D, self.D//2, 2, [self.batch_size, s4, s4, self.D//2], name='de_layer1'))
    x = tf.nn.relu(self.deconv2d(x, 4, self.D//2, self.D//2, 2, [self.batch_size, s2, s2, self.D//2], name='de_layer2'))
    x = self.deconv2d(x, 4, self.D//2, self.input_channel, 2, 
        [self.batch_size, self.output_width, self.output_width, self.input_channel], name='de_layer3')
    x_recons = tf.nn.tanh(x)
    return x, x_recons    

  def residual(self, x, name):
    conv1 = tf.layers.conv2d(x, self.D, (3, 3), (1, 1), activation=tf.nn.leaky_relu, padding='same', name="%s_1" % name)
    conv2 = tf.layers.conv2d(conv1, self.D, (1, 1), (1, 1), activation=None, padding='same', name="%s_2" % name)
    return tf.nn.leaky_relu(conv2 + x)

  def reconstruct(self, x_images):
    feed_dict = {self.x_images: x_images}
    return self.sess.run(self.x_recons, feed_dict=feed_dict)

  def generate(self, z, batch_size):
    feed_dict= {self.z: z, self.batch_size: batch_size}
    return self.sess.run(self.x_recons, feed_dict=feed_dict)

  def train(self, x_images):
    feed_dict = {self.x_images: x_images}
    return self.sess.run([self.train_op, self.loss, self.rec_loss, self.vq, self.commit,
        self.global_step, self.summary], feed_dict=feed_dict)

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
    return self.sess.run(self.x_recons, feed_dict=feed_dict)

  def debug(self, x_images):
    feed_dict = {self.x_images: x_images}
    return self.sess.run([self.embed, self.k], feed_dict=feed_dict)