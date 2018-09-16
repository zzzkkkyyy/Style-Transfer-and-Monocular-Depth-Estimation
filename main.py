import tensorflow as tf
import numpy as np
import os, random, glob
import scipy.misc as misc

batch_size = 1
path = '/data'
width = 128
height = 384

def ResnetBlock(input_layer, kernel_size=3, padding='same', is_dropout=True):
    input_channels = input_layer.get_shape()[-1]
    layer = tf.layers.conv2d(input_layer, filters=input_channels, kernel_size=kernel_size, padding=padding)
    layer = tf.nn.leaky_relu(tf.layers.batch_normalization(layer))
    if is_dropout is True:
        layer = tf.nn.dropout(layer, keep_prob=0.5)
    layer = tf.layers.conv2d(input_layer, filters=input_channels, kernel_size=kernel_size, padding=padding)
    layer = tf.layers.batch_normalization(layer)
    if layer.get_shape() != input_layer.get_shape():
        raise ValueError('In ResnetBlock input shape is not the same as output.')
    return input_layer + layer

def ResnetGenerator(input_layer, output_channels, is_training=True, is_back=False):
    num_downsampling = 2
    num_resnet_block = 9
    layer = tf.layers.conv2d(input_layer, filters=64, kernel_size=5, padding='same')
    layer = tf.nn.leaky_relu(tf.layers.batch_normalization(layer))
    for i in range(num_downsampling):
        output_channel = 2 ** (i + 6)
        layer = DownSampling(layer, output_channel)
    for i in range(num_resnet_block):
        layer = ResnetBlock(layer, is_dropout=is_training)
    for i in range(num_downsampling):
        output_channel = 2 ** (5 + num_downsampling - i)
        layer = UpSampling(layer, output_channel, is_dropout=is_training)
    layer = tf.layers.conv2d_transpose(layer, filters=output_channels, kernel_size=5, padding='same')
    if is_back is True:
        #layer = tf.nn.tanh(tf.layers.batch_normalization(layer))
        layer = (tf.nn.sigmoid(tf.layers.batch_normalization(layer)) - 0.5) / 0.5
    else:
        layer = (tf.nn.sigmoid(tf.layers.batch_normalization(layer)) - 0.5) / 0.5
    return layer

def DownSampling(input_layer, output_channels, kernel_size=3, strides=2, is_leaky_relu=True, is_norm=False):
    layer = tf.layers.conv2d(input_layer, filters=output_channels, kernel_size=kernel_size, strides=strides, padding='same')
    if is_norm is True:
        layer = tf.layers.batch_normalization(layer)
    if is_leaky_relu is True:
        layer = tf.nn.leaky_relu(layer)
    return layer

def UnetSubModules(input_layer, kernel_size=3, is_leaky_relu=True):
    layer = tf.layers.conv2d(input_layer, filters=input_layer.get_shape()[-1], kernel_size=kernel_size, strides=1, padding='same')
    if is_leaky_relu is True:
        layer = tf.nn.leaky_relu(layer)
    return layer

def UpSampling(input_layer, output_channels, kernel_size=3, is_norm=True, is_relu=True, is_dropout=True):
    layer = tf.layers.conv2d_transpose(input_layer, filters=output_channels, kernel_size=kernel_size, strides=2, padding='same')
    if is_norm is True:
        layer = tf.layers.batch_normalization(layer)
    if is_relu is True:
        layer = tf.nn.leaky_relu(layer)
    if is_dropout is True:
        layer = tf.nn.dropout(layer, keep_prob=0.5)
    return layer

def UnetGenerator(input_layer, output_channels, is_training=True):
    layer = input_layer
    num_downsampling = 4
    num_unetblock = 2
    for i in range(num_downsampling):
        output_channel = 2 ** (i + 6)
        layer = DownSampling(layer, output_channels)
    for i in range(num_unetblock):
        layer = UnetSubModules(layer)
    for i in range(num_downsampling - 1):
        output_channel = 2 ** (8 - i)
        layer = UpSampling(layer, output_channel, is_dropout=is_training)
    layer = UpSampling(layer, output_channels, is_dropout=is_training)
    return layer

def NLayerDiscriminator(input_layer):
    layer = input_layer
    for i in range(3):
        output_channels = 2 ** (i + 6)
        layer = DownSampling(layer, output_channels=output_channels, is_norm=True)
    layer = DownSampling(layer, output_channels=output_channels, strides=1, is_norm=True)
    layer = tf.layers.conv2d(layer, filters=1, kernel_size=3, padding='same')
    return tf.nn.sigmoid(layer)

def PixelDiscriminator(input_layer):
    layer = DownSampling(input_layer, output_channels=64, kernel_size=1, strides=1)
    layer = DownSampling(layer, output_channels=64 * 2, kernel_size=1, strides=1, is_norm=True)
    layer = DownSampling(input_layer, output_channels=1, kernel_size=1, strides=1, is_leaky_relu=False)
    return tf.nn.sigmoid(layer)

class BatchDatasetReader:
    def __init__(self, batch_size, path):
        self.data_dict = {'training': {'road': [], 'lane': []}, 'testing': []}
        self.length_dict = {'training': {}, 'testing': 0}
        self.batch_size = batch_size
        self.current_road_index = 0
        self.current_lane_index = 0
        self.test_index = 0
        
        item_list = ['training', 'testing']
        for item in item_list:
            p_image = os.path.join(path, item, 'image_2', '*.png')
            p_depth = os.path.join(path, item, 'gt_image_2', '*.png')
            image_list = []
            image_list.extend(glob.glob(p_image))
            if item == 'training':
                depth_list = []
                depth_list.extend(glob.glob(p_depth))
            if not image_list or not depth_list:
                print("file not found.")
                return None

            if item == 'training':
                flag = False
                for f in depth_list:
                    if flag is False:
                        flag = True
                        image = misc.imread(f)
                        print("kitti dataset shape:", image.shape)
                    t = 'road'
                    filename = os.path.split(f)[-1]
                    imagename = ''
                    if filename.find('road') != -1:
                        index_start = filename.find('road_')
                        index_end = index_start + len('road_')
                        imagename = filename[: index_start] + filename[index_end: ]
                    else:
                        index_start = filename.find('lane_')
                        index_end = index_start + len('lane_')
                        imagename = filename[: index_start] + filename[index_end: ]
                        t = 'lane'
                    image_path = os.path.join(path, item, 'image_2', imagename)
                    if image_path not in image_list:
                        print(image_path)
                        break
                    else:
                        im = misc.imresize(misc.imread(image_path), size=(width, height))
                        de = misc.imresize(misc.imread(f), size=(width, height))
                        im = (im - np.min(im)) / (np.max(im) - np.min(im))
                        de = (de - np.min(de)) / (np.max(de) - np.min(de))
                        im = (im - 0.5) / 0.5
                        de = (de - 0.5) / 0.5
                        self.data_dict[item][t].append([im, de])
            else:
                """
                for f in image_list:
                    im = misc.imresize(misc.imread(f), size=(width, height))
                    self.data_dict['testing'].append(im)
                """

        print('training road dataset length:', len(self.data_dict['training']['road']))
        print('training lane dataset length:', len(self.data_dict['training']['lane']))
        #print('testing dataset length:', len(self.data_dict['testing']))

        self.length_dict['training']['road'] = len(self.data_dict['training']['road'])
        self.length_dict['training']['lane'] = len(self.data_dict['training']['lane'])
        #self.length_dict['testing'] = len(self.data_dict['testing'])
        
        random.shuffle(self.data_dict['training']['road'])
        random.shuffle(self.data_dict['training']['lane'])
        #random.shuffle(self.data_dict['testing'])

    def ReadNextBatch(self, is_training=True):
        t = 'testing'
        if is_training is True:
            t = 'training'
        if t == 'training':
            if self.current_lane_index + self.batch_size >= self.length_dict[t]['lane']:
                random.shuffle(self.data_dict[t]['lane'])
                self.current_lane_index = 0
            if self.current_road_index + self.batch_size >= self.length_dict[t]['road']:
                random.shuffle(self.data_dict[t]['road'])
                self.current_road_index = 0
            
            road_image = [item[0] for item in self.data_dict[t]['road'][self.current_road_index: self.current_road_index + self.batch_size]]
            road_depth = [item[1] for item in self.data_dict[t]['road'][self.current_road_index: self.current_road_index + self.batch_size]]
            lane_image = [item[0] for item in self.data_dict[t]['lane'][self.current_lane_index: self.current_lane_index + self.batch_size]]
            lane_depth = [item[1] for item in self.data_dict[t]['lane'][self.current_lane_index: self.current_lane_index + self.batch_size]]
            l = np.array([lane_image, road_image, road_depth, lane_depth])
            self.current_lane_index += self.batch_size
            return l
        else:
            if self.test_index + self.batch_size >= self.length_dict[t]:
                random.shuffle(self.data_dict[t])
                self.test_index = 0
            self.test_index += self.batch_size
            return self.data_dict[t][self.test_index - self.batch_size: self.test_index]

def main(argv=None):
    print("init reading...")
    reader = BatchDatasetReader(batch_size=batch_size, path=path)
    
    print("constructing network...")
    x_true = tf.placeholder(tf.float32, shape=(batch_size, width, height, 3), name='input')
    y_true = tf.placeholder(tf.float32, shape=(batch_size, width, height, 3), name='converted')
    z_true = tf.placeholder(tf.float32, shape=(batch_size, width, height, 1), name='output')
    is_training = tf.placeholder(tf.bool, shape=(), name='is_training')
    tf.summary.image('x_true', x_true, 2)
    tf.summary.image('y_true', y_true, 2)

    with tf.variable_scope('y_generator'):
        y_fake = ResnetGenerator(x_true, output_channels=3, is_training=is_training)
        tf.summary.image('y_generate', y_fake, 2)
    with tf.variable_scope('x_generator', reuse=tf.AUTO_REUSE):
        x_fake = ResnetGenerator(y_true, output_channels=3, is_training=is_training, is_back=True)
        x_fake_true = ResnetGenerator(y_fake, output_channels=3, is_training=is_training, is_back=True)
        tf.summary.image('x_fake_true', x_fake_true, 2)
    with tf.variable_scope('y_generator', reuse=True):
        y_fake_true = ResnetGenerator(x_fake, output_channels=3, is_training=is_training)
    with tf.variable_scope('x_discriminator', reuse=tf.AUTO_REUSE):
        x_truth_loss = -tf.log(tf.reduce_mean(NLayerDiscriminator(x_true)))
        x_fake_loss = -tf.log(1 - tf.reduce_mean(NLayerDiscriminator(x_fake)))
        x_loss = x_truth_loss + 1 * x_fake_loss
        x_cycle_loss = tf.reduce_mean(tf.abs(x_true - x_fake_true))
    with tf.variable_scope('y_discriminator', reuse=tf.AUTO_REUSE):
        y_truth_loss = -tf.log(tf.reduce_mean(NLayerDiscriminator(y_true)))
        y_fake_loss = -tf.log(1 - tf.reduce_mean(NLayerDiscriminator(y_fake)))
        y_loss = y_truth_loss + 1 * y_fake_loss
        y_cycle_loss = tf.reduce_mean(tf.abs(y_true - y_fake_true))
    
    total_loss = x_loss + y_loss + 10 * (x_cycle_loss + y_cycle_loss)

    with tf.variable_scope('z_generator'):
        z_fake = UnetGenerator(y_true, output_channels=1, is_training=is_training)
        #tf.summary.image('z_generate', z_fake)
    with tf.variable_scope('z_discriminator', reuse=tf.AUTO_REUSE):
        z_truth_loss = tf.log(PixelDiscriminator(z_true))
        z_fake_loss = tf.log(1 - PixelDiscriminator(z_fake))
        z_loss = tf.reduce_mean(z_truth_loss + z_fake_loss)
        
    trainable_variable = {'x_generator': [], 'x_discriminator': [], 'y_generator': [], 'y_discriminator': [], 'z_generator': [], 'z_discriminator': []}
    for item in tf.trainable_variables():
        for label in trainable_variable.keys():
            if item.name.startswith(label):
                trainable_variable[label].append(item)
    
    #domain_transfer_var = trainable_variable['x_generator'] + trainable_variable['x_discriminator'] + trainable_variable['y_generator'] + trainable_variable['y_discriminator']
    domain_transfer_generate_var = trainable_variable['x_generator'] + trainable_variable['y_generator']
    domain_transfer_discriminator_var = trainable_variable['x_discriminator'] + trainable_variable['y_discriminator']
    estimation_var = trainable_variable['z_generator'] + trainable_variable['z_discriminator']
    del trainable_variable
    
    #domain_transfer_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3, beta1=0.5).minimize(total_loss, var_list=domain_transfer_var)
    domain_transfer_g_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5).minimize(total_loss, var_list=domain_transfer_generate_var)
    domain_transfer_d_optimizer = tf.train.AdamOptimizer(learning_rate=1e-5, beta1=0.5).minimize(total_loss, var_list=domain_transfer_discriminator_var)
    #domain_transfer_d_optimizer_2 = tf.train.AdamOptimizer(learning_rate=1e-5, beta1=0.5).minimize(total_loss, var_list=domain_transfer_discriminator_var)
    estimation_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3, beta1=0.5).minimize(z_loss, var_list=estimation_var)

    print("begin running...")
    merged = tf.summary.merge_all()
    sess = tf.Session()
    summary_writer = tf.summary.FileWriter("/output/board", sess.graph)
    sess.run(tf.global_variables_initializer())
    """
    print("training discriminator...")
    for i in range(50):
        x_image, y_image, _, _ = reader.ReadNextBatch()
        y_t_loss, y_f_loss, loss, _ = sess.run([y_truth_loss, y_fake_loss, total_loss, domain_transfer_d_optimizer], feed_dict={x_true: x_image, y_true: y_image, is_training: True})
        if (i + 1) % 5 == 0:
            print("iteration:", i + 1, "------------> domain transfer training loss:", y_t_loss, y_f_loss, loss)
    """
    print("training domain transfer generator and discriminator...")
    for i in range(1000):
        x_image, y_image, _, _ = reader.ReadNextBatch()
        loss, y_generate, _ = sess.run([total_loss, y_fake, domain_transfer_g_optimizer], feed_dict={x_true: x_image, y_true: y_image, is_training: True})
        if (i + 1) % 5 == 0:
            y_t_loss, y_f_loss, summary, loss, y_generate, _ = sess.run([y_truth_loss, y_fake_loss, merged, total_loss, y_fake, domain_transfer_d_optimizer], feed_dict={x_true: x_image, y_true: y_image, is_training: True})
            print("iteration:", i + 1, "------------> domain transfer training loss:", y_t_loss, y_f_loss, loss)
            summary_writer.add_summary(summary, i + 1)
    
    print("training depth generator...")
    for i in range(0):
        _, y_image, z_image, _ = reader.ReadNextBatch()
        summary, loss, z_generate, _ = sess.run([merged, z_loss, z_fake, estimation_optimizer], feed_dict={y_true: y_image, z_true: z_image, is_training: True})
        if (i + 1) % 5 == 0:
            print("iteration:", i + 1, "------------> depth estimation training loss:", loss)
            summary_writer.add_summary(summary, i + 1)
    return

if __name__ == "__main__":
    tf.app.run()