import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.layers import (
    InputLayer,
    Conv2D,
    MaxPooling2D,
    ZeroPadding2D,
    LocallyConnected2D,
    Dense,
    LeakyReLU,
    Flatten,
    BatchNormalization,
    Dropout
)

def add_conv_or_local(is_conv, model, block, block_index):

  filters = block['filters']
  kernel_size = block['size']
  strides = block['stride']

  # add padding
  padding = block['pad']
  if padding == 1:
    if is_conv:
      pad = 'same'
    else:
      pad_size = (kernel_size - 1) // 2
      pad = 'valid'
      model.add(ZeroPadding2D(pad_size, name=f'pad_{block_index}'))
  else:
    pad = 'valid'
  
  # check batch norm
  try:
      batch_norm = (block['batch_normalize'] == 1)
  except:
      batch_norm = False
  
  use_bias = not batch_norm

  # add Conv2D or LocallyConnected2D
  if is_conv:
    layer_class = Conv2D
    layer_name = f'conv_{block_index}'
  else:
    pad = 'valid'
    layer_class = LocallyConnected2D
    layer_name = f'local_{block_index}'

  model.add(layer_class(filters, kernel_size, strides, use_bias=use_bias,
                        padding=pad, name=layer_name))
  
  # add batch norm
  if batch_norm:
    model.add(BatchNormalization(name=f'batchnorm_{block_index}'))
  
  # add activation function
  activation = block['activation']
  if activation == 'leaky':
    model.add(LeakyReLU(alpha=0.1, name=f'leaky_{block_index}'))

class YOLODetection(tf.keras.layers.Layer):
  def __init__(self, s, b, c, softmax_class_probs=True, sigmoid_box_conf=True,
               sigmoid_box_coords=True, *args, **kwargs):
    super(YOLODetection, self).__init__(*args, **kwargs)
    self.grid_size = s
    self.num_boxes = b
    self.num_classes = c
    self.softmax_class_probs = softmax_class_probs
    self.sigmoid_box_conf = sigmoid_box_conf
    self.sigmoid_box_coords = sigmoid_box_coords

  def call(self, x):
    batch_size = K.shape(x)[0]
    s = self.grid_size
    b = self.num_boxes
    c = self.num_classes

    class_probs_end = s*s*c
    box_confidence_end = class_probs_end + s*s*b

    class_probs = K.reshape(x[:, :class_probs_end], (batch_size, s, s, c))#tf.reshape(x[:, :class_probs_end], [batch_size, s, s, c])

    if self.softmax_class_probs:
      class_probs = tf.nn.softmax(class_probs)
    
    box_confidence = tf.reshape(x[:, class_probs_end:box_confidence_end], [batch_size, s, s, b])
    if self.sigmoid_box_conf:
      box_confidence = tf.nn.sigmoid(box_confidence)
    
    box_coords = tf.reshape(x[:, box_confidence_end:], [batch_size, s, s, b*4])
    if self.sigmoid_box_coords:
      box_coords = tf.nn.sigmoid(box_coords)

    outputs = tf.concat([class_probs, box_confidence, box_coords], axis=-1)

    return outputs

def create_model_from_config(config):
  """
  Create a tensorflow.keras model using config parsed
  from parse_config()
  """

  model = tf.keras.models.Sequential()
  block_index = 0

  for name, block in config:
    if name == 'net':
      input_shape = (block['height'], block['width'], block['channels'])
      model.add(InputLayer(input_shape=input_shape, name='input_0'))
    elif name == 'convolutional':
      add_conv_or_local(True, model, block, block_index)
    elif name == 'maxpool':
      size = block['size']
      strides = block['stride']
      pooling = MaxPooling2D(size, strides, name=f'maxpooling_{block_index}')
      model.add(pooling)
    elif name == 'local':
      add_conv_or_local(False, model, block, block_index)
    elif name == 'dropout':
      rate = block['probability']
      drop_out = Dropout(rate, name=f'dropout_{block_index}')
      model.add(drop_out)
    elif name == 'connected':
      units = block['output']
      activation = block['activation']

      flatten = Flatten(name=f'flatten_{block_index}')
      dense = Dense(units, name=f'fullyconnected_{block_index}')

      model.add(flatten)
      model.add(dense)

      if activation == 'leaky':
        leaky = LeakyReLU(alpha=0.1, name=f'leakyrelu_{block_index}')
        model.add(leaky)

    elif name == 'detection':
      s = block['side']
      b = block['num']
      c = block['classes']
      
      detection = YOLODetection(s, b, c, name=f'detection_{block_index}')
      model.add(detection)

    block_index += 1
  
  return model