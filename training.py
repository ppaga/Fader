import os, sys, pprint, time

import numpy as np
import tensorflow as tf
import tensorlayer as tl

import model
from data_utils import CelebA

def to_categorical(x):
    #assume x is of the form [batchsize, 1]
    y = 1-x
    z = tf.concat([x,y],axis=1)
    return z

def attributes_loss(x,y):
    #assume x is of the form [batchsize, len_attributes, 2], 
    #while y is of the form [batchsize, len_attributes]
    n = y.shape.as_list()[-1]
    x_split = tf.split(x,n, axis=1)
    y_split = tf.split(y,n, axis=1)
    x_split = [tf.squeeze(t) for t in x_split]
    y_split = [to_categorical(t) for t in y_split]
    splits = list(zip(x_split, y_split))
    loss = 0
    for logits, labels in splits:
        loss += tf.losses.softmax_cross_entropy(labels, logits)
    
    return loss
    
def train(run_name, updates, attributes = ['Young', 'Eyeglasses','Male'], image_size = 256, c_dim=3, resume = False, resume_from=0, save_step = int(1e2)):

    shape = [image_size, image_size, c_dim]
    
    input_image = tf.placeholder(tf.float32, shape=[None,]+shape)
    attributes_input = tf.placeholder(tf.int32, shape = [None,len(attributes)])
    
    encoder = model.Encoder(input_image, name='Encoder')

    attributes_for_decoder = tf.cast(tf.expand_dims(tf.expand_dims(attributes_input, 1),1), tf.float32)    
    decoder = model.Decoder(encoder, attributes_for_decoder, name='Decoder')
    decoded_image = decoder.outputs

    discriminator = model.Discriminator(encoder, len(attributes), name='Discriminator')
    logits = discriminator.outputs

    global_step = tf.Variable(resume_from, name = 'global_step', trainable=False)
    increment = tf.assign(global_step, global_step+1)
    #lambda_e, according to the original paper, increases linearly over 500 000 time steps up to a maximum of 1e-4
    lambda_E = tf.minimum(2e-10*tf.cast(global_step, tf.float32), tf.constant(1e-4))
    
    AE_loss = tf.losses.mean_squared_error(decoded_image, input_image)
    disc_loss = attributes_loss(logits, attributes_input)
    enc_loss = attributes_loss(logits, 1-attributes_input)

    AE_loss = AE_loss + lambda_E*enc_loss

    # list of generator and discriminator variables for use by the optimizers
    enc_vars = tl.layers.get_variables_with_name('Encoder', True, False)
    dec_vars = tl.layers.get_variables_with_name('Decoder', True, False)
    disc_vars = tl.layers.get_variables_with_name('Discriminator', True, False)

    learning_rate = 2*1e-3
    beta1 = 0.5
    
    AE_op = tf.train.AdamOptimizer(learning_rate, beta1 = beta1).minimize(AE_loss, var_list = enc_vars+dec_vars)
    AE_op = tf.group(AE_op, increment)
    disc_op = tf.train.AdamOptimizer(learning_rate, beta1 = beta1).minimize(disc_loss, var_list = disc_vars)
    
    with tf.name_scope('loss_summaries/'):
        tf.summary.scalar('encoder_loss', enc_loss)
        tf.summary.scalar('AE_loss', AE_loss)
        tf.summary.scalar('discriminator_loss',disc_loss)
        tf.summary.scalar('lambda_E', lambda_E)
        tf.summary.histogram('logits', logits)
    merged = tf.summary.merge_all()

    saver = tf.train.Saver(enc_vars+dec_vars+disc_vars)

    with tf.Session() as sess:
        tl.layers.initialize_global_variables(sess)

        # generates the various folders in case they don't already exist, starting with the run directory
        run_dir = './runs/'+run_name+'/'
        checkpoint_dir = run_dir+'checkpoints/'
        logs_dir = run_dir+'logs/'

        tl.files.exists_or_mkdir(run_dir)
        tl.files.exists_or_mkdir(checkpoint_dir)
        tl.files.exists_or_mkdir(logs_dir)

        #creates the tensorboard log writer
        writer = tf.summary.FileWriter(logs_dir, graph=tf.get_default_graph())

        # creates the image pipeline
        imshape = [image_size, image_size]
        data_generator = CelebA(imshape = imshape)

        ##========================= TRAIN MODELS ================================##

        if resume: 
            print('loading model from '+checkpoint_dir)
            saver.restore(sess,tf.train.latest_checkpoint(checkpoint_dir))
            print('model loaded')
        start_time = time.time()
        for update in range(resume_from, updates):
            # generate low and high-resolution image batch:
            batch = data_generator.batch(attributes_list = attributes)
            # train the network and generate summary statistics for tensorboard
            feed_dict_disc = {input_image : batch[0], attributes_input : batch[1]}
            feed_dict_disc.update(discriminator.all_drop)
            _ = sess.run(disc_op,feed_dict = feed_dict_disc)
            
            batch = data_generator.batch(attributes_list = attributes)
            feed_dict_AE = {input_image : batch[0], attributes_input : batch[1]}
            feed_dict_AE.update(discriminator.all_drop)
            _ = sess.run(AE_op,feed_dict = feed_dict_AE)
            
            print("update [%3d/%3d]" % (update, updates))
            
            if update>0 and (np.mod(update, save_step) == 0):
                summary = sess.run(merged, feed_dict=feed_dict_disc)
                writer.add_summary(summary, update)
                # save current network parameters
                saver.save(sess, checkpoint_dir, global_step = update)
                print("checkpoint saved at "+checkpoint_dir)
        
        saver.save(sess, pretrain_dir, global_step = update)
        print("model fully trained and saved at "+checkpoint_dir)




















