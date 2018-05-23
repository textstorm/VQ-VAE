
import tensorflow as tf
import collections

Datasets = collections.namedtuple('Datasets', ['train', 'test'])

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

  def batch(self):
    return self.sess.run(self._batch_ops)

  def __del__(self):
    print ' [*] DiskImageData: stop threads and close session!'
    self.coord.request_stop()
    self.coord.join(self.threads)
    self.sess.close()

  @property
  def num_examples(self):
    return self._num_examples