
import tensorflow as tf
import collections
import scipy.misc
import numpy as np

def preprocess_fn(img):
  crop_size = 108
  re_size = 64
  img = tf.image.crop_to_bounding_box(img, (218 - crop_size) // 2, (178 - crop_size) // 2, crop_size, crop_size)
  img = tf.image.resize_images(img, [re_size, re_size], method=tf.image.ResizeMethod.BICUBIC)
  img = tf.subtract(img, tf.reduce_min(img)) / tf.subtract(tf.reduce_max(img), tf.reduce_min(img)) * 2 - 1
  return img

def disk_image_batch(image_paths, batch_size, shape, shuffle=True, num_threads=16,
                     min_after_dequeue=100, allow_smaller_final_batch=False, scope=None):
  with tf.name_scope(scope, 'disk_image_batch'):
    data_num = len(image_paths)
    _, img = tf.WholeFileReader().read(tf.train.string_input_producer(image_paths, shuffle=shuffle, capacity=data_num))
    img = tf.image.decode_image(img)

    img.set_shape(shape)
    img = preprocess_fn(img)

    if shuffle:
      capacity = min_after_dequeue + (num_threads + 1) * batch_size
      img_batch = tf.train.shuffle_batch([img], batch_size=batch_size, capacity=capacity, 
                                         min_after_dequeue=min_after_dequeue, num_threads=num_threads,
                                         allow_smaller_final_batch=allow_smaller_final_batch)
    else:
      img_batch = tf.train.batch([img], batch_size=batch_size, allow_smaller_final_batch=allow_smaller_final_batch)

    return img_batch, data_num

class DiskImageData:
  def __init__(self, sess, image_paths, batch_size, shape, shuffle=True, num_threads=16,
               min_after_dequeue=100, allow_smaller_final_batch=False, scope=None):

    self._batch_ops, self._data_num = disk_image_batch(image_paths, batch_size, shape, shuffle, num_threads,
                                                       min_after_dequeue, allow_smaller_final_batch, scope)

    print ' [*] DiskImageData: create session!'
    self.sess = sess
    self.coord = tf.train.Coordinator()
    self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

    self._num_examples = len(image_paths)

  def __len__(self):
    return self._data_num

  def next_batch(self):
    return self.sess.run(self._batch_ops)

  def __del__(self):
    print ' [*] DiskImageData: stop threads and close session!'
    self.coord.request_stop()
    self.coord.join(self.threads)
    self.sess.close()

  @property
  def num_examples(self):
    return self._num_examples

def get_config_proto(log_device_placement=False, allow_soft_placement=True):
  config_proto = tf.ConfigProto(
      log_device_placement=log_device_placement,
      allow_soft_placement=allow_soft_placement)
  config_proto.gpu_options.allow_growth = True
  return config_proto

def to_range(images, min_value=0.0, max_value=1.0, dtype=None):
  assert \
    np.min(images) >= -1.0 - 1e-5 and np.max(images) <= 1.0 + 1e-5 \
    and (images.dtype == np.float32 or images.dtype == np.float64), \
    'The input images should be float64(32) and in the range of [-1.0, 1.0]!'
  if dtype is None:
    dtype = images.dtype
  return ((images + 1.) / 2. * (max_value - min_value) + min_value).astype(dtype)

def save_images(images, size, image_path):
  """Save mutiple images into one single image.
  Parameters
  -----------
  images : numpy array [batch, w, h, c]
  size : list of two int, row and column number.
      number of images should be equal or less than size[0] * size[1]
  image_path : string.
  Examples
  ---------
  >>> images = np.random.rand(64, 100, 100, 3)
  >>> utils.save_images(images, [8, 8], 'temp.png')
  """
  def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j*h:j*h+h, i*w:i*w+w, :] = image
    return img

  def imsave(images, size, path):
    return scipy.misc.imsave(path, to_range(merge(images, size), 0, 255, np.uint8))

  assert len(images) <= size[0] * size[1], "number of images should be equal or less than size[0] * size[1] {}".format(len(images))
  return imsave(images, size, image_path)