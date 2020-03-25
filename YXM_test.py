# -*-coding:utf-8 -*-
import numpy as np
import pickle
import matplotlib.pyplot as plt
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras import utils
from sklearn.model_selection import train_test_split
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import concatenate
from keras.layers import Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.models import load_model



#读取数据：
np.random.seed(23)
def load_data(file_data):
    with open(file_data, mode='rb') as f:
        train = pickle.load(f)
        # print(train)
        # print(type(train))
        X = train['features']
        y = train['labels']

        return X, y

#展示数据：
def show_random_samples(X_train, y_train, n_classes):
    # show a random sample from each class of the traffic sign dataset
    rows, cols = 4, 12
    fig, ax_array = plt.subplots(rows, cols)    #fig：画板 ；ax_array：子图和集合，用二维矩阵表示
    plt.suptitle('Random Samples (one per class)')
    for class_idx, ax in enumerate(ax_array.ravel()):
        if class_idx < n_classes:
            # show a random image of the current class
            cur_X = X_train[y_train == class_idx]
            cur_img = cur_X[np.random.randint(len(cur_X))]
            ax.imshow(cur_img)                       #X : array_like, shape (n, m) or (n, m, 3) or (n, m, 4)
            ax.set_title('{:02d}'.format(class_idx))
        else:
            ax.axis('off')
    # hide both x and y ticks
    plt.setp([a.get_xticklabels() for a in ax_array.ravel()], visible=False)  #设置
    plt.setp([a.get_yticklabels() for a in ax_array.ravel()], visible=False)
    plt.draw()


def show_classes_distribution(n_classes, y_train, n_train):
    # bar-chart of classes distribution
    train_distribution = np.zeros(n_classes)
    for c in range(n_classes):
        train_distribution[c] = np.sum(y_train == c) / n_train
    fig, ax = plt.subplots()
    col_width = 1
    bar_train = ax.bar(np.arange(n_classes), train_distribution, width=col_width, color='r')
    ax.set_ylabel('Percentage')
    ax.set_xlabel('Class Label')
    ax.set_title('Distribution')
    ax.set_xticks(np.arange(0, n_classes, 5) + col_width)
    ax.set_xticklabels(['{:02d}'.format(c) for c in range(0, n_classes, 5)])
    plt.show()

def show_image(X, y):

    n_train = X.shape[0]
    image_shape = X[0].shape
    n_classes = np.unique(y).shape[0]
    print('训练数据集的形状', X.shape)
    print("训练数据集的数据个数=", n_train)
    print("图像尺寸  =", image_shape)
    print("类别数量 =", n_classes)

    show_random_samples(X, y, n_classes)
    show_classes_distribution(n_classes, y, n_train)

#数据预处理
def get_mean_std_img(X):
    # convert from RGB to YUV：色彩空间转换
    X = np.array([np.expand_dims(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)[:, :, 0], 2) for img_rgb in X])

    # adjust image contrast:直方图均衡化
    X = np.array([np.expand_dims(cv2.equalizeHist(np.uint8(img_yuv)), 2) for img_yuv in X])

    X = np.float32(X)
    mean_train = np.mean(X, axis=0)
    std_train = np.std(X, axis=0)

    return mean_train, std_train

def preprocess_features(X, mean_train, std_train):
    # convert from RGB to YUV：色彩空间转换   Y表示明亮度（Luminance、Luma），U和V则是色度、浓度（Chrominance、Chroma）
        # 解释：交通指示牌的辨别，不是依靠颜色来辨别，而是主要依靠交通指示牌中的几何形状；而且一般交通指示牌为了让肉眼看的更加清楚，往往红底蓝字，亮度比较大
        # 所以我们可以不考虑颜色，而考虑交通指示牌图片中不同位置亮度的差别这样一个几何形状来辨别交通指示牌
        # 这样有两个好处：1、相当于把数据量减少了 ，把原来三个通道的图片变成一个通道的图片,处理的数据量减少2、把不想要的色彩信息去掉，让模型更加集中考虑几何形状等关键信息
    X = np.array([np.expand_dims(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)[:, :, 0], 2) for img_rgb in X])

    # adjust image contrast:直方图均衡化
         #图像的直方图是对图像对比度效果上的一种处理，旨在使得图像整体效果均匀，黑与白之间的各个像素级之间的点更均匀一点。
         #通过这种方法，亮度可以更好地在直方图上分布。这样就可以用于增强局部的对比度而不影响整体的对比度，直方图均衡化通过有效地扩展常用的亮度来实现这种功能。
    X = np.array([np.expand_dims(cv2.equalizeHist(np.uint8(img_yuv)), 2) for img_yuv in X])  #np.uint8(img_yuv):从RGB to YUV的转换不一定会转换为整数而是可能变成浮点数和实数，先将转化为uint8

    #standardize features
        #减去平均值的原因：我们系统识别标识符，主要是看图片的差异，为了突出差异，我们可以减去平均值；
        #除以标准差的原因：原来的图片中存在变化比较大的地方，那么该部分的方差也会比较大，在网络训练中，权值也会比较大，那么网络就容易关注局部而忽视整体；同时整个网络容易陷入局部最优解，收敛速度会降低。除以方差后，分布更加均匀。
    X = np.float32(X)
    X -= mean_train
    X /= std_train

    return X

#数据增强
def show_samples_from_generator(image_datagen, X_train, y_train):
    # take a random image from the training set  （工具函数）
    img_rgb = X_train[0]
    # plot the original image
    plt.figure(figsize=(1, 1))
    plt.imshow(img_rgb)
    plt.title('Example of RGB image (class = {})'.format(y_train[0]))
    plt.show()

    # plot some randomly augmented images
    rows, cols = 4, 10
    fig, ax_array = plt.subplots(rows, cols)
    for ax in ax_array.ravel():
        augmented_img, _ = image_datagen.flow(np.expand_dims(img_rgb, 0), y_train[0:1]).next()    # next() 返回迭代器的下一个项目
        ax.imshow(np.uint8(np.squeeze(augmented_img)))
    plt.setp([a.get_xticklabels() for a in ax_array.ravel()], visible=False)
    plt.setp([a.get_yticklabels() for a in ax_array.ravel()], visible=False)
    plt.suptitle('Random examples of data augmentation (starting from the previous image)')
    plt.show()




def get_image_generator():

    # create the generator to perform online data augmentation：数据增强，数据变换
        #keras.ImageDataGenerator：通过实时数据增强生成张量图像数据批次，数据将不断循环
             #实时数据：预先不产生，在训练数据的产生，再在内存中计算以免占用空间，也免除了从硬盘中读取数据的过程
        #ImageDataGenerator里面的参数是随机组合生成图片的

    image_datagen = ImageDataGenerator(rotation_range=15., #随机15度以内旋转，产生新的数据
                                       zoom_range=0.2,  #随机缩放——缩小或者放大，缩放比值是0.2
                                       width_shift_range=0.1, #随机宽这个方向的挪动（左右）
                                       height_shift_range=0.1  #随机高这个方向的挪动（上下 ）
                                       )
    return image_datagen



def show_image_generator_effect():
    # 数据增强演示：
    X_train, y_train = load_data('./traffic-signs-data/train.p')

    # Number of examples
    n_train = X_train.shape[0]

    # What's the shape of an traffic sign image?
    image_shape = X_train[0].shape  #四维数据即批处理，X_train[0].shape：第一个图片样本的形状

    # How many classes?
    n_classes = np.unique(y_train).shape[0]

    print("训练数据集的数据个数=", n_train)
    print("图像尺寸  =", image_shape)
    print("类别数量 =", n_classes)

    image_generator = get_image_generator()
    # print(image_generator[1:2])
    print('image_generator:', type(image_generator))
    # print(image_generator.shape)
    show_samples_from_generator(image_generator, X_train, y_train)

#训练模型：
def get_model(dropout_rate = 0.0):
    input_shape = [32, 32,1]

    # 第一条路径(下面的路径)：高清的特征图，看的图片的细节更多
    input = Input(shape=input_shape)
    cv2D_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input)
    pool_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(cv2D_1)        #strides=2与strides=(2, 2)等价
    dropout_1 = Dropout(dropout_rate)(pool_1)    #为了防止过拟合自己添加的一个Dropout层
    flatten_1 = Flatten()(dropout_1)

    # 第二条路径（上面的路径）：不是高清的特征图，看的图片的整体更多
    cv2d_2 = Conv2D(64, (3, 3), padding='same', activation='relu')(dropout_1)
    pool_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(cv2d_2)
    cv2d_3 = Conv2D(64, (3, 3), padding='same', activation='relu')(pool_2)
    dropout_2 = Dropout(dropout_rate)(cv2d_3)
    flatten_2 = Flatten()(dropout_2)

    #两条路线合并
    con = concatenate([flatten_1, flatten_2])
    dense1 = Dense(64, activation='relu')(con)
    output = Dense(43, activation='softmax')(dense1)
    model = Model(inputs=input, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()  #使用keras构建深度学习模型，我们会通过model.summary()输出模型各层的参数状况

    return model

def train(model, image_datagen, x_train, y_train, x_validation, y_validation):

    filepath = "./history/weights1.best.hdf5"
    checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')


    cb_logger = CSVLogger(filename='./history/train.csv', separator=',', append=True)

    callbacks_list = [checkpoint, cb_logger]
    image_datagen.fit(x_train)

    history = model.fit_generator(image_datagen.flow(x_train, y_train, batch_size=128),  #batch_size=128:每次从原数据中取128个数据来进行数据增强
                                  steps_per_epoch=5000,  #每一个epochs有5000步，每一步取128图像作为训练
                                  epochs=8,
                                  validation_data=(x_validation, y_validation),
                                  callbacks=callbacks_list,   #训练集训练几遍，每训练一遍运行结束运行一次checkpoint
                                  verbose=1
                                  )
    # list all data in history
    print(history.history.keys())   #history为字典，存入形式是history_dict；history分别将accuracy，val_accuracy,loss,val_loss提取出来，加了val前缀的是每个epoche所得到的模型测试集得到的结果
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # 把训练过程数据存在这个文件之中，以此来查看数据的变化，是够存在过拟合，欠拟合
    with open('./history/trainHistoryDict1.p', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
    return history


def train_model():

    #读取数据
    X, y = load_data('./traffic-signs-data/train.p')
    n_train = X.shape[0]
    image_shape = X[0].shape
    n_classes = np.unique(y).shape[0]
    print("Number of training examples =", n_train)
    print("Image data shape  =", image_shape)
    print("Number of classes =", n_classes)

    #数据预处理
    mean_train, std_train = get_mean_std_img(X)
    X = preprocess_features(X, mean_train=mean_train, std_train=std_train)

    #转化独热编码
    y = utils.to_categorical(y, n_classes)

    #数据集与验证集
    VAL_RATIO = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=VAL_RATIO, random_state=0)

    #数据增强
    X_generator = get_image_generator()

    #构造模型
    model = get_model(0.0)

    #训练模型
    train(model, X_generator, X_train, y_train, X_test, y_test)



#评估模型
#通过测试数据，放入到evaluate函数中，来查看准确率为多少，以此来决定整个项目的最终的准确率
def evaluate_accuracy(model, X_test, y_test):

    score = model.evaluate(X_test, y_test, verbose=1)
    accuracy = score[1]

    return accuracy

def evaluate():
    X_test, y_test = load_data('./traffic-signs-data/test.p')
    mean_X, std_X = get_mean_std_img(X_test)
    X_test = preprocess_features(X_test, mean_X, std_X)

    y_test = utils.to_categorical(y_test, 43)

    # model = load_model('./history/weights1.best.hdf5')
    model = load_model('./history/train.csv')

    accuracy = evaluate_accuracy(model, X_test, y_test)
    print(accuracy)



if __name__ == "__main__":

    # 读取并展示数据
    # X, y = load_data('./traffic-signs-data/train.p')
    # print(X.shape)
    # show_image(X, y)

    #数据预处理
    # mean_X, std_X = get_mean_std_img(X)
    # X = preprocess_features(X, mean_X, std_X)
    # print(type(X))
    # print('预处理之后', X.shape)
    # X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))
    # print('形状格式化', X.shape)
    # show_image(X, y)


    #数据增强：
    # show_image_generator_effect()

    #训练网络模型
    # train_model()

    #评估模型
    # evaluate()
    pass






#补充学习：
#补充
#（1）
    # x = np.array([[1, 2, 5], [2, 3, 5], [3, 4, 5], [2, 3, 6]])
    # print(x.shape)  # 结果： (4, 3)  # 输出数组的行和列数
    # print(x.shape[0])  # 结果：4    # 只输出行数
    # print(x.shape[1])  # 结果：3    # 只输出列数
#（2）

#(3)

#(4)
    # cv2.equalizeHist(img)
    # 将要均衡化的原图像【要求是灰度图像】作为参数传入，则返回值即为均衡化后的图像。
#(5)
    # keras.utils.to_categorical方法:  to_categorical(y, num_classes=None, dtype='float32')
    # 将整型标签转为onehot。y为int数组，num_classes为标签类别总数，大于max(y)（标签从0开始的）。
    # 返回：如果num_classes=None，返回len(y) * [max(y)+1]（维度，m*n表示m行n列矩阵，下同），否则为len(y) * num_classes。
#(6)
    # keras.layers.core.Dense(
    #     units,  # 代表该层的输出维度
    #     activation=None,  # 激活函数.但是默认 liner
    #     use_bias=True,  # 是否使用b
    #     kernel_initializer='glorot_uniform',  # 初始化w权重，keras/initializers.py
    #     bias_initializer='zeros',  # 初始化b权重
    #     kernel_regularizer=None,  # 施加在权重w上的正则项,keras/regularizer.py
    #     bias_regularizer=None,  # 施加在偏置向量b上的正则项
    #     activity_regularizer=None,  # 施加在输出上的正则项
    #     kernel_constraint=None,  # 施加在权重w上的约束项
    #     bias_constraint=None  # 施加在偏置b上的约束项
    # )
#(7)ModelCheckpoint
# keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
# 该回调函数将在每个epoch后保存模型到filepath
    # filepath可以是格式化的字符串，里面的占位符将会被epoch值和传入on_epoch_end的logs关键字所填入
    # 例如，filepath若为weights.{epoch:02d-{val_loss:.2f}}.hdf5，则会生成对应epoch和验证集loss的多个文件。
    # 参数
    # filename：字符串，保存模型的路径
    # monitor：需要监视的值
    # verbose：信息展示模式，0或1（verbose = 0 为不在标准输出流输出日志信息；verbose = 1 为输出进度条记录；verbose = 2 为每个epoch输出一行记录）
    # save_best_only：当设置为True时，将只保存在验证集上性能最好的模型
    # mode：‘auto’，‘min’，‘max’之一，在save_best_only=True时决定性能最佳模型的评判准则，例如，当监测值为val_acc时，模式应为max，当检测值为val_loss时，模式应为min。在auto模式下，评价准则由被监测值的名字自动推断。
    # save_weights_only：若设置为True，则只保存模型权重，否则将保存整个模型（包括模型结构，配置信息等）
    # period：CheckPoint之间的间隔的epoch数。

#(8)Keras.ImageDataGenerator()
    # ImageDataGenerator()是keras.preprocessing.image模块中的图片生成器，同时也可以在batch中对数据进行增强，扩充数据集大小，增强模型的泛化能力。比如进行旋转，变形，归一化等等。
    # keras.preprocessing.image.ImageDataGenerator(featurewise_center=False, samplewise_center=False,
                                                  #featurewise_std_normalization=False,
                                                  #samplewise_std_normalization=False, zca_whitening=False,
                                                  #zca_epsilon=1e-06, rotation_range=0.0, width_shift_range=0.0,
                                                  #height_shift_range=0.0, brightness_range=None, shear_range=0.0,
                                                  #zoom_range=0.0, channel_shift_range=0.0, fill_mode='nearest', cval=0.0,
                                                  #horizontal_flip=False, vertical_flip=False, rescale=None,
                                                  #preprocessing_function=None, data_format=None, validation_split=0.0)
    # 常用参数:
        # featurewise_center: Boolean. 对输入的图片每个通道减去每个通道对应均值。
        # samplewise_center: Boolan. 每张图片减去样本均值, 使得每个样本均值为0。
        # featurewise_std_normalization(): Boolean()
        # samplewise_std_normalization(): Boolean()
        # zca_epsilon(): Default 12-6
        # zca_whitening: Boolean. 去除样本之间的相关性
        # rotation_range(): 旋转范围
        # width_shift_range(): 水平平移范围
        # height_shift_range(): 垂直平移范围
        # shear_range(): float, 透视变换的范围
        # zoom_range(): 缩放范围
        # fill_mode: 填充模式, constant, nearest, reflect
        # cval: fill_mode == 'constant'的时候填充值
        # horizontal_flip(): 水平反转
        # vertical_flip(): 垂直翻转
        # preprocessing_function(): user提供的处理函数
        # data_format(): channels_first或者channels_last
        # validation_split(): 多少数据用于验证集
    #常用方法：
    # 方法:
        # apply_transform(x, transform_parameters): 根据参数对x进行变换
        # fit(x, augment=False, rounds=1,
        #     seed=None): 将生成器用于数据x, 从数据x中获得样本的统计参数, 只有featurewise_center, featurewise_std_normalization或者zca_whitening为True才需要
        # flow(x, y=None, batch_size=32, shuffle=True, sample_weight=None, seed=None, save_to_dir=None, save_prefix='',
        #      save_format='png', subset=None) ):按batch_size大小从x, y生成增强数据
        # flow_from_directory()
        # 从路径生成增强数据, 和flow方法相比最大的优点在于不用一次将所有的数据读入内存当中, 这样减小内存压力，这样不会发生OOM，血的教训。
        # get_random_transform(img_shape, seed=None): 返回包含随机图像变换参数的字典
        # random_transform(x, seed=None): 进行随机图像变换, 通过设置seed可以达到同步变换。
        # standardize(x): 对x进行归一化

        #data generators：由于数据量太大，不能把数据一次性全部都放进内存中处理，可以使用generators生成器批量处理数据，节省内存，加快处理速度
#(9)：keras 两种训练模型方式fit和fit_generator(节省内存)
    #fit参数详情
        # keras.models.fit(
        # self,
        # x=None, #训练数据
        # y=None, #训练数据label标签
        # batch_size=None, #每经过多少个sample更新一次权重，defult 32
        # epochs=1, #训练的轮数epochs
        # verbose=1, #0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
        # callbacks=None,#list，list中的元素为keras.callbacks.Callback对象，在训练过程中会调用list中的回调函数
        # validation_split=0., #浮点数0-1，将训练集中的一部分比例作为验证集，然后下面的验证集validation_data将不会起到作用
        # validation_data=None, #验证集
        # shuffle=True, #布尔值和字符串，如果为布尔值，表示是否在每一次epoch训练前随机打乱输入样本的顺序，如果为"batch"，为处理HDF5数据
        # class_weight=None, #dict,分类问题的时候，有的类别可能需要额外关注，分错的时候给的惩罚会比较大，所以权重会调高，体现在损失函数上面
        # sample_weight=None, #array,和输入样本对等长度,对输入的每个特征+个权值，如果是时序的数据，则采用(samples，sequence_length)的矩阵
        # initial_epoch=0, #如果之前做了训练，则可以从指定的epoch开始训练
        # steps_per_epoch=None, #将一个epoch分为多少个steps，也就是划分一个batch_size多大，比如steps_per_epoch=10，则就是将训练集分为10份，不能和batch_size共同使用
        # validation_steps=None, #当steps_per_epoch被启用的时候才有用，验证集的batch_size; ##当steps_per_epoch被启用的时候才有用，验证集的batch_size;当validation_data为生成器时，本参数指定验证集的生成器返回次数
                                     #仅当 validation_data 是一个生成器时才可用。 在停止前 generator 生成的总步数（样本批数）。 对于 Sequence，它是可选的：如果未指定，将使用 len(generator) 作为步数。
        # **kwargs #用于和后端交互
        # )
        # 返回的是一个History对象，可以通过History.history来查看训练过程，loss值等等
#(10)  model.evaluate 和 model.predict 的区别
        # model.evaluate:输入数据和标签,输出损失和精确度.   用于评估
        # model.predict:输入测试数据,输出预测结果    用于预测
#(11)python 中  numpy.flatten()()   和 numpy.ravel()
    # 两者的本质都是想把多维的数组降为1维。区别在于numpy.flatten()
    # 返回一份拷贝，对数据更改时不会影响原来的数组，而numpy.ravel()
    # 则返回视图，对数据更改时会影响原来的数组。
# (12)# Python enumerate() 函数
# 描述:enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中
    #语法：enumerate(sequence, [start=0])
    #参数：参数
        # sequence -- 一个序列、迭代器或其他支持迭代对象。
        # start -- 下标起始位置。
        # 实例：
        # >>>seasons = ['Spring', 'Summer', 'Fall', 'Winter']
        # >>> list(enumerate(seasons))
        # [(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
        # >>> list(enumerate(seasons, start=1))       # 下标从 1 开始
        # [(1, 'Spring'), (2, 'Summer'), (3, 'Fall'), (4, 'Winter')]
#(13)numpy.random.randint用法:numpy.random.randint(low, high=None, size=None, dtype='l')
    # 函数的作用是，返回一个随机整型数，范围从低（包括）到高（不包括），即[low, high)。
    # 如果没有写参数high的值，则返回[0, low)的值。
    # 参数如下：
        # low: int   生成的数值最低要大于等于low。（hign = None时，生成的数值要在[0, low)区间内）
        # high: int(可选)  如果使用这个值，则生成的数值在[low, high)区间。
        # size: int or tuple of  ints(可选)   输出随机数的尺寸，比如size = (m * n * k)   则输出同规模即m * n * k个随机数。默认是None的，仅仅返回满足要求的单一随机数。
        # dtype: dtype(可选)：想要输出的格式。如int64、int等等
        # 输出： out: int or ndarray of ints   返回一个随机数或随机数数组
#（14）关于数据的读取与模型的保存和读取
    #keras：.hdf5文件
        # model.save('my_model.h5')  # 创建 HDF5 文件 'my_model.h5'
        # del model  # 删除现有模型

        # 返回一个编译好的模型
        # 与之前那个相同
        # model = load_model('my_model.h5')

        # 如果要加载的模型包含自定义层或其他自定义类或函数，则可以通过custom_objects参数将它们传递给加载机制：
            # 假设你的模型包含一个 AttentionLayer 类的实例
            # model = load_model('my_model.h5', custom_objects={'AttentionLayer': AttentionLayer})
            # model = load_model('.h5', custom_objects={'r2_score_dfy': r2_score_dfy})
            # model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=[r2_score_dfy])  #自定义评估函数
    #pick中：.p文件
        # pickle.load(f)   读取文件
        # pickle.dump(history.history, file_pi)  保存模型
