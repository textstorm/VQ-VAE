
#1.read mnist dataset
import gzip
import numpy as np
import collections
import scipy.misc
import tensorflow as tf

Dataset = collections.namedtuple('Dataset', ['data', 'target'])
Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])

def _read32(bytestream):
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]

def extract_images(f):
  print 'Extracting', f.name
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError('Invalid magic number %d in MNIST image file: %s' % (magic, f.name))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = np.frombuffer(buf, dtype=np.uint8)
    data = data.reshape(num_images, rows, cols, 1)
    return data

def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

def extract_labels(f, one_hot=False, num_classes=10):
  print 'Extracting', f.name
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                       (magic, f.name))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = np.frombuffer(buf, dtype=np.uint8)
    if one_hot:
      return dense_to_one_hot(labels, num_classes)
    return labels

def read_data_sets(train_dir, one_hot=False, dtype=np.float32, reshape=True,
                   validation_size=5000):

  TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
  TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
  TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
  TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

  local_file = train_dir + '/' + TRAIN_IMAGES
  with open(local_file, 'rb') as f:
    train_images = extract_images(f)

  local_file = train_dir + '/' + TRAIN_LABELS

  with open(local_file, 'rb') as f:
    train_labels = extract_labels(f, one_hot=one_hot)

  local_file = train_dir + '/' + TEST_IMAGES
                                   
  with open(local_file, 'rb') as f:
    test_images = extract_images(f)

  local_file = train_dir + '/' + TEST_LABELS

  with open(local_file, 'rb') as f:
    test_labels = extract_labels(f, one_hot=one_hot)

  if not 0 <= validation_size <= len(train_images):
    raise ValueError(
        'Validation size should be between 0 and {}. Received: {}.'
        .format(len(train_images), validation_size))

  validation_images = train_images[:validation_size]
  validation_labels = train_labels[:validation_size]
  train_images = train_images[validation_size:]
  train_labels = train_labels[validation_size:]

  train = DataSet(train_images, train_labels, dtype=dtype, reshape=reshape)
  validation = DataSet(validation_images, validation_labels, dtype=dtype, reshape=reshape)
  test = DataSet(test_images, test_labels, dtype=dtype, reshape=reshape)

  return Datasets(train=train, validation=validation, test=test)

def load_mnist(train_dir='MNIST-data'):
  return read_data_sets(train_dir)

class DataSet(object):

  def __init__(self, images, labels, one_hot=False, dtype=np.float32,
               reshape=True):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.
    """
    if dtype not in (np.uint8, np.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' % dtype)

    assert images.shape[0] == labels.shape[0], (
        'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
    self._num_examples = images.shape[0]

    # Convert shape from [num examples, rows, columns, depth]
    # to [num examples, rows*columns] (assuming depth == 1)
    if reshape:
      assert images.shape[3] == 1
      images = images.reshape(images.shape[0], images.shape[1] * images.shape[2])
    if dtype == np.float32:
      # Convert from [0, 255] -> [0.0, 1.0].
      images = images.astype(np.float32)
      images = np.multiply(images, 1.0 / 255.0)

    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]

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
    return scipy.misc.imsave(path, merge(images, size))

  assert len(images) <= size[0] * size[1], "number of images should be equal or less than size[0] * size[1] {}".format(len(images))
  return imsave(images, size, image_path)

def get_config_proto(log_device_placement=False, allow_soft_placement=True):
  config_proto = tf.ConfigProto(
      log_device_placement=log_device_placement,
      allow_soft_placement=allow_soft_placement)
  config_proto.gpu_options.allow_growth = True
  return config_proto