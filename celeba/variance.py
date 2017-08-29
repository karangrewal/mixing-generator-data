#!/usr/bin/env python

"""
VRAL with meta-discriminators on CelebA.
"""

import numpy as np
from lasagne.layers import get_all_params, get_output, get_all_layers
from lasagne.updates import adam
from theano import function
import theano.tensor as T
import os

from data import get_data
from model import discriminator, generator
from real_fake import real_fake_discriminator

print('starting')

# BATCH NORM PARAMS
BN_D = False
BN_G = False

# HYPERPARAMS
ADAM_BETA1 = 0.1
ADAM_BETA2 = 0.999
ADAM_EPSILON = 3e-6
ADAM_LR = 0.0001
BATCH_SIZE = 64
DIM_C = 3
DIM_X = 64
DIM_Y = 64
DIM_Z = 128
EPOCHS = 100
ITERS_F = 1
ITERS_R = 1
ITERS_D = 1

REAL_MEAN = 1.0
FAKE_MEAN = -1.0

if __name__ == '__main__':
    if BN_D and BN_G:
        out_dir = '/u/grewalka/celeba/variance/%d_%d_%d_1/' % (ITERS_F, ITERS_R, ITERS_D)
    elif BN_D:
        out_dir = '/u/grewalka/celeba/variance/bn_d/%d_%d_%d_1/' % (ITERS_F, ITERS_R, ITERS_D)
    elif BN_G:
        out_dir = '/u/grewalka/celeba/variance/bn_g/%d_%d_%d_1/' % (ITERS_F, ITERS_R, ITERS_D)
    else:
        out_dir = '/u/grewalka/celeba/variance/nbn/%d_%d_%d_1/' % (ITERS_F, ITERS_R, ITERS_D)
    
    with open(os.path.join(out_dir, 'out.log'), 'w+') as f:
        f.write('Variance-maximizing GAN\n')
        f.write('F iters: {}, R iters: {}, D iters: {}\n'.format(ITERS_F, ITERS_R, ITERS_D))
        f.write('BN on D: {}\tBN on G: {}'.format(BN_D, BN_G))

    X = T.tensor4()
    z = T.fmatrix()

    D, G = discriminator(X, use_batch_norm=False), generator(z, use_batch_norm=False)

    y_real = get_output(D)
    X_fake = get_output(G)
    y_fake = get_output(D, X_fake)

    # Real and fake discriminators
    F, R = real_fake_discriminator(y_fake), real_fake_discriminator(y_real)

    # Samples from N(fake mean,1) and N(real mean,1)
    v_0 = T.fmatrix()
    v_1 = T.fmatrix()

    # Outputs of real and fake discriminators
    r_real = get_output(R, v_1)
    r_fake = get_output(R)
    f_real = get_output(F, v_0)
    f_fake = get_output(F)

    # Loss functions
    F_loss = (T.nnet.softplus(-f_real) + T.nnet.softplus(-f_fake) + f_fake).mean()
    R_loss = (T.nnet.softplus(-r_real) + T.nnet.softplus(-r_fake) + r_fake).mean()
    D_loss = (T.nnet.softplus(-f_fake) + T.nnet.softplus(-r_fake)).mean()
    G_loss = 0.5 * ((y_fake - 1) ** 2).mean()

    # Updates to be performed during training
    updates_F = adam(loss_or_grads=F_loss,params=get_all_params(F, trainable=True),
        learning_rate=ADAM_LR,beta1=ADAM_BETA1,beta2=ADAM_BETA2,epsilon=ADAM_EPSILON)

    updates_R = adam(loss_or_grads=R_loss,params=get_all_params(R, trainable=True),
        learning_rate=ADAM_LR,beta1=ADAM_BETA1,beta2=ADAM_BETA2,epsilon=ADAM_EPSILON)

    updates_D = adam(loss_or_grads=D_loss,params=get_all_params(D, trainable=True),
        learning_rate=ADAM_LR,beta1=ADAM_BETA1,beta2=ADAM_BETA2,epsilon=ADAM_EPSILON)

    updates_G = adam(loss_or_grads=G_loss,params=get_all_params(G, trainable=True),
        learning_rate=ADAM_LR,beta1=ADAM_BETA1,beta2=ADAM_BETA2,epsilon=ADAM_EPSILON)


    train_F = function([v_0, z], outputs=F_loss, updates=updates_F, allow_input_downcast=True)
    train_R = function([v_1, X], outputs=R_loss, updates=updates_R, allow_input_downcast=True)
    train_D = function([X, z], outputs=D_loss, updates=updates_D, allow_input_downcast=True)
    train_G = function([z], outputs=G_loss, updates=updates_G, allow_input_downcast=True)

    # Gradient Norms
    G_loss_grad = T.grad(G_loss, X_fake)
    G_loss_grad_norm_value = function([z],outputs=(G_loss_grad**2).sum(axis=(1,2,3)).mean())
    D_loss_grad = T.grad(D_loss, X_fake)
    D_loss_grad_norm_value = function([X, z],outputs=(D_loss_grad**2).sum(axis=(1,2,3)).mean())

    # Sampling functions
    generate = function([z], outputs=X_fake)
    D_out_R = function([X], outputs=y_real)
    D_out_F = function([z], outputs=y_fake)
    
    z_samples = np.float32(np.random.normal(size=(BATCH_SIZE, DIM_Z)))
    
    # Load data
    stream, num_examples = get_data(BATCH_SIZE)

    print('VRAL with meta-discriminators, training ratio = {}:{}:{}:1'.format(ITERS_F, ITERS_R, ITERS_D))
    print('Output files will be placed in: {}'.format(out_dir))

    for epoch in range(EPOCHS):
        print('Starting Epoch {}/{} ...'.format(epoch+1, EPOCHS))

        G_loss_grad_norms = np.zeros(shape=(num_examples / BATCH_SIZE))
        D_loss_grad_norms = np.zeros(shape=(num_examples / BATCH_SIZE, ITERS_D))
        D_samples = np.zeros(shape=((num_examples / BATCH_SIZE)*BATCH_SIZE , 2))
        
        # Training
        for i in range(num_examples / BATCH_SIZE):
            
            iterator = stream.get_epoch_iterator()
            for k in range(ITERS_D):
            
                # Train fake data discriminator
                v_0_i = np.float32(np.random.normal(loc=FAKE_MEAN,size=(BATCH_SIZE,1)))
                z_i = np.float32(np.random.normal(size=(BATCH_SIZE,DIM_Z)))
                train_F(v_0_i, z_i)
                
                # Train real data discriminator
                v_1_i = np.float32(np.random.normal(loc=REAL_MEAN,size=(BATCH_SIZE,1)))
                x_i = iterator.next()[0]
                train_R(v_1_i, x_i)
                
                # Train discriminator
                z_i = np.float32(np.random.normal(size=(BATCH_SIZE,DIM_Z)))
                train_D(x_i, z_i)
                D_loss_grad_norms[i,k] = D_loss_grad_norm_value(x_i, z_i)
            
            # train generator
            z_i = np.float32(np.random.normal(size=(BATCH_SIZE,DIM_Z)))
            train_G(z_i)
            G_loss_grad_norms[i] = G_loss_grad_norm_value(z_i)

        # Generate samples from G
        x_samples = generate(z_samples)
        with open(os.path.join(out_dir, 'x_samples_%d.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, x_samples)

        # Sample from D
        iterator = stream.get_epoch_iterator()
        for i in range(num_examples / BATCH_SIZE):
            x_i = iterator.next()[0]
            z_i = np.float32(np.random.normal(size=(BATCH_SIZE,DIM_Z)))
            D_samples[i*BATCH_SIZE:(i+1)*BATCH_SIZE,0:1] = D_out_R(x_i)
            D_samples[i*BATCH_SIZE:(i+1)*BATCH_SIZE,1:2] = D_out_F(z_i)

        # Save Results
        with open(os.path.join(out_dir, 'D_samples_%d.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, D_samples)
        
        # Save gradient norms
        with open(os.path.join(out_dir, 'D_loss_grad_norms_%d.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, D_loss_grad_norms)
        with open(os.path.join(out_dir, 'G_loss_grad_norms_%d.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, G_loss_grad_norms)
        
        # Save model
        if epoch % 5 == 4:
            D_params = get_all_params(get_all_layers(D))
            with open(os.path.join(out_dir, 'discriminator_model_{}.npz'.format(epoch+1)), 'w+') as f:
                np.savez(f, D_params)
            
            G_params = get_all_params(get_all_layers(G))
            with open(os.path.join(out_dir, 'generator_model_{}.npz'.format(epoch+1)), 'w+') as f:
                np.savez(f, G_params)


