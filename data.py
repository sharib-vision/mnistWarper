import numpy as np
from keras.utils import Sequence

class FashionMNISTDataHandler(object):
  """
    Members :
      is_train - Options for sampling
      path - MNIST data path
      data - a list of np.array w/ shape [batch_size, 28, 28, 1]
  """
  def __init__(self, path, is_train):
    self.is_train = is_train
    self.path = path
    self.data = self._get_data()

  def _get_data(self):
    from tensorflow.contrib.learn.python.learn.datasets.base \
      import maybe_download
    from tensorflow.contrib.learn.python.learn.datasets.mnist \
      import extract_images, extract_labels

    if self.is_train:
      IMAGES = 'train-images-idx3-ubyte.gz'
      LABELS = 'train-labels-idx1-ubyte.gz'
    else :
      print('using test dataset..')
      IMAGES = 't10k-images-idx3-ubyte.gz'
      LABELS = 't10k-labels-idx1-ubyte.gz'
    SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'

    local_file = maybe_download(IMAGES, self.path, SOURCE_URL)
    with open(local_file, 'rb') as f:
      images = extract_images(f)
    local_file = maybe_download(LABELS, self.path, SOURCE_URL)
    with open(local_file, 'rb') as f:
      labels = extract_labels(f, one_hot=False)

    values, counts = np.unique(labels, return_counts=True)

    data = []
    for i in range(10):
      label = values[i]
      count = counts[i]
      arr = np.empty([count, 1, 28, 28], dtype=np.float32)
      data.append(arr)

    l_iter = [0]*10
    for i in range(labels.shape[0]):
      label = labels[i]
      data[label][l_iter[label]] = np.reshape(images[i], [1,28,28]) / 255.
      l_iter[label] += 1
      
    self.data = data
    self.l_iter = l_iter

    return data

  def sample_pair(self, batch_size, label=None):
    label = np.random.randint(10) if label is None else label
    images = self.data[label]
    
    choice1 = np.random.choice(images.shape[0], batch_size)
    choice2 = np.random.choice(images.shape[0], batch_size)
    x = images[choice1]
    y = images[choice2]

    return x, y

  def data_generation(self, batch_size, label = None):
    if batch_size == 0:
        batch_size = max(self.l_iter)
    self.channel = 1
    label = np.random.randint(10) if label is None else label
#    label = 5 if label is None else label
    images = self.data[label]
    img_size = 28
    if self.channel==3:
        x = np.zeros((batch_size, img_size, img_size, 3), dtype=np.float32)
        y = np.zeros((batch_size, img_size, img_size, 3), dtype=np.float32)
    else:
        #            print('channel is:', self.channel)
        x = np.zeros((batch_size, img_size, img_size, 1), dtype=np.float32)
        y = np.zeros((batch_size, img_size, img_size, 1), dtype=np.float32)
        z = np.zeros((batch_size, img_size, img_size, 2), dtype=np.float32)
        
    sample_id = 0 
    while True:
        choice1 = np.random.choice(images.shape[0], 1)
        choice2 = np.random.choice(images.shape[0], 1)
        
        x[sample_id] = images[choice1]
        y[sample_id] = images[choice2]
        sample_id += 1
        
        
        if sample_id == batch_size:
            return x, y
        
        
class MNISTDataHandler(object):
  """
    Members :
      is_train - Options for sampling
      path - MNIST data path
      data - a list of np.array w/ shape [batch_size, 28, 28, 1]
  """
  def __init__(self, path, is_train):
    self.is_train = is_train
    self.path = path
    self.data = self._get_data()

  def _get_data(self):
    from tensorflow.contrib.learn.python.learn.datasets.base \
      import maybe_download
    from tensorflow.contrib.learn.python.learn.datasets.mnist \
      import extract_images, extract_labels

    if self.is_train:
      IMAGES = 'train-images-idx3-ubyte.gz'
      LABELS = 'train-labels-idx1-ubyte.gz'
    else :
      print('using test dataset..')
      IMAGES = 't10k-images-idx3-ubyte.gz'
      LABELS = 't10k-labels-idx1-ubyte.gz'
    SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'

    local_file = maybe_download(IMAGES, self.path, SOURCE_URL)
    with open(local_file, 'rb') as f:
      images = extract_images(f)
    local_file = maybe_download(LABELS, self.path, SOURCE_URL)
    with open(local_file, 'rb') as f:
      labels = extract_labels(f, one_hot=False)

    values, counts = np.unique(labels, return_counts=True)

    data = []
    for i in range(10):
      label = values[i]
      count = counts[i]
      arr = np.empty([count, 1, 28, 28], dtype=np.float32)
      data.append(arr)

    l_iter = [0]*10
    for i in range(labels.shape[0]):
      label = labels[i]
      data[label][l_iter[label]] = np.reshape(images[i], [1,28,28]) / 255.
      l_iter[label] += 1
      
    self.data = data
    self.l_iter = l_iter

    return data

  def sample_pair(self, batch_size, label=None):
#    label = 7 if label is None else label
    label = np.random.randint(10) if label is None else label
    images = self.data[label]
    choice1 = np.random.choice(images.shape[0], batch_size)
    choice2 = np.random.choice(images.shape[0], batch_size)
    x = images[choice1]
    y = images[choice2]

    return x, y

  def data_generation(self, batch_size, label = None):
    if batch_size == 0:
        batch_size = max(self.l_iter)
    self.channel = 1
    label = np.random.randint(10) if label is None else label
#    label = 5 if label is None else label
    images = self.data[label]
    img_size = 28
    if self.channel==3:
        x = np.zeros((batch_size, img_size, img_size, 3), dtype=np.float32)
        y = np.zeros((batch_size, img_size, img_size, 3), dtype=np.float32)
    else:
        #            print('channel is:', self.channel)
        x = np.zeros((batch_size, img_size, img_size, 1), dtype=np.float32)
        y = np.zeros((batch_size, img_size, img_size, 1), dtype=np.float32)
        z = np.zeros((batch_size, img_size, img_size, 2), dtype=np.float32)
        
    sample_id = 0 
    while True:
        choice1 = np.random.choice(images.shape[0], 1)
        choice2 = np.random.choice(images.shape[0], 1)
        x[sample_id] = images[choice1]
        y[sample_id] = images[choice2]
        sample_id += 1
        
        
        if sample_id == batch_size:
            return x, y
        
