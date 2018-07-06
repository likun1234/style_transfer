import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import numpy as np
import tensorflow as tf

model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")
#print(model)

content_image = scipy.misc.imread("images/hong_kong_2.jpg")
imshow(content_image)
plt.show()

def compute_content_cost(a_C, a_G):

    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    a_C_unrolled = tf.reshape(a_C, shape=[m, n_H * n_W, n_C])
    a_G_unrolled = tf.reshape(a_G, shape=[m, n_H * n_W, n_C])
    J_content = 1. / (4 * n_H * n_W * n_C) * tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled)))

    return J_content

style_image = scipy.misc.imread("images/starry_night.jpg")
imshow(style_image)
plt.show()

def gram_matrix(A):
    GA = tf.matmul(A, tf.transpose(A))
    return GA

def compute_layer_style_cost(a_S, a_G):

    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    a_S = tf.reshape(tf.transpose(a_S, perm=[0, 3, 2, 1]), shape=[n_C, n_H * n_W])
    a_G = tf.reshape(tf.transpose(a_G, perm=[0, 3, 2, 1]), shape=[n_C, n_H * n_W])

    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)
    J_style_layer =  1. / (4 * n_H * n_H * n_W * n_W * n_C * n_C) * tf.reduce_sum(tf.square(tf.subtract(GS, GG)))

    return J_style_layer

STYLE_LAYERS = [('conv1_1', 0.2), ('conv2_1', 0.2), ('conv3_1', 0.2), ('conv4_1', 0.2), ('conv5_1', 0.2)]

def compute_style_cost(model, STYLE_LAYERS):
    J_style = 0
    for layer_name, coeff in STYLE_LAYERS:
        out = model[layer_name]
        a_S = sess.run(out)
        a_G = out
        J_style_layer = compute_layer_style_cost(a_S, a_G)
        J_style += coeff * J_style_layer

    return J_style

def total_cost(J_content, J_style, alpha = 10, beta = 40):
    J = alpha * J_content + beta * J_style
    return J

tf.reset_default_graph()
sess = tf.InteractiveSession()

content_image = scipy.misc.imread('images/hong_kong.jpg')
content_image = reshape_and_normalize_image(content_image)
style_image = scipy.misc.imread('images/starry_night.jpg')
style_image = reshape_and_normalize_image(style_image)
generate_image = generate_noise_image(content_image)
imshow(generate_image[0])

model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")
sess.run(model['input'].assign(content_image))
out = model['conv4_2']
a_C = sess.run(out)
a_G = out
J_content = compute_content_cost(a_C, a_G)
sess.run(model['input'].assign(style_image))
J_style = compute_style_cost(model, STYLE_LAYERS)
J = total_cost(J_content, J_style, 10, 40)

optimizer = tf.train.AdamOptimizer(1.0)
train_step = optimizer.minimize(J)

def model_nn(sess, input_image, num_iterations = 1000):
    sess.run(tf.global_variables_initializer())
    sess.run(model['input'].assign(input_image))
    for i in range(num_iterations):

        sess.run(train_step)
        generate_image = sess.run(model['input'])
        if i % 20 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))
            save_image("output/" + str(i) + ".png", generate_image)

    save_image("output/generated_image.jpg", generate_image)

    return generate_image

model_nn(sess, generate_image)
