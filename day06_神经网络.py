"""
Created on 2019/3/12 8:58
@File:day06_神经网络.py
@author: coderwangson
"""
"#codeing=utf-8"
import tensorflow as tf
import cv2 as cv

from tensorflow.examples.tutorials.mnist import input_data
# python day06_神经网络.py --is_train=False
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_boolean("is_train",True,"is train")

def fc():
    mnist = input_data.read_data_sets("./data/mnist",one_hot=True)
    # 1、建立数据占位符（在作用域下）
    with tf.variable_scope("data"):
        x = tf.placeholder(tf.float32,[None,784])
        y_true = tf.placeholder(tf.int32,[None,10])

    # 2、建立全连接层的神经网络 w b （作用域下）
    with tf.variable_scope("model"):
        # w、b
        w = tf.Variable(tf.random_normal([784,10],dtype = tf.float32))
        b = tf.Variable(tf.constant(1.0,shape = [10]))
        # 计算预测值
        y_pre = tf.matmul(x,w)+b
    # 3、计算sotfmx以及交叉熵并求平均
    with tf.variable_scope("loss"):
        # tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pre) 求取的是每个样本的误差
        # 所以最后需要进行取平均
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pre))

    # 4、梯度下降
    with tf.variable_scope("optimizer"):
        train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    # 5、计算准确率
    with tf.variable_scope("acc"):
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(y_true,1),tf.arg_max(y_pre,1)),tf.float32))

    # 添加到摘要
    tf.summary.scalar("loss",loss)
    tf.summary.scalar("acc",acc)

    # 高纬度
    tf.summary.histogram("w",w)
    tf.summary.histogram("b",b)
    # 图片
    # tf.summary.image("img",tf.reshape(x[0],[1,28,28,1]))
    merge = tf.summary.merge_all()
    # writer = tf.summary.FileWriter("./log",tf.get_default_graph())


    # 开启会话训练（变量初始化）
    init_op = tf.global_variables_initializer()

    # 保存模型
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init_op)
        writer = tf.summary.FileWriter("./log", sess.graph)
        if FLAGS.is_train:
            # 多轮 并且喂入数据 mnist_x,mnist_y = mnist.train.next_batch(50)
            for i in range(500):
                mnist_x,mnist_y = mnist.train.next_batch(50)
                _,acc_v,loss_v,summary,y_pre_v = sess.run([train_op,acc,loss,merge,y_pre],feed_dict={x:mnist_x,y_true:mnist_y})

                writer.add_summary(summary,i)
                print("the acc is %f"%acc_v)
                print("the loss is %f"%loss_v)
                if i%10==0:

                    image =tf.reshape(mnist_x[0], [28, 28, 1]).eval()
                    pre_label = tf.arg_max(y_pre_v,1)[0].eval()
                    image_summary = tf.summary.image("img_label"+str(pre_label), tf.reshape(image,[1,28,28,1]))
                    writer.add_summary(sess.run(image_summary))
            saver.save(sess,"./ckpt/mnist")
        else:
            saver.restore(sess,"./ckpt/mnist")
            mnist_x,mnist_y = mnist.test.next_batch(50)
            image_label = tf.arg_max(mnist_y,1).eval()
            pre_v = sess.run(y_pre,feed_dict={x:mnist_x,y_true:mnist_y})
            pre_label = tf.arg_max(pre_v,1).eval()
            for i in range(image_label.shape[0]):
                print("the %dth image,the true label is %d,the pre_label is %d"%(i,image_label[i],pre_label[i]))

    return None


if __name__ == '__main__':
    fc()
