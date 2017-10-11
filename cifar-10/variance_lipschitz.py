#!/usr/bin/env python

"""
VRAL: forcing slope of tangent between real and fake samples to follow a 
gaussian on CIFAR-10.
"""

print('starting')

import numpy as np
from lasagne.layers import get_all_params, get_output, get_all_layers
from lasagne.layers import batch_norm, Conv2DLayer, DenseLayer, InputLayer, ReshapeLayer, TransposedConv2DLayer
from lasagne.nonlinearities import LeakyRectify, tanh
from lasagne.updates import adam
from theano import function
import theano.tensor as T
import os

from data import get_data
from real_fake import real_fake_discriminator

# HIDDEN UNITS
DIM_H = 128

# BATCH NORM PARAMS
BN_D = True
BN_G = True

# HYPERPARAMS
ADAM_BETA1 = 0.1
ADAM_BETA2 = 0.999
ADAM_EPSILON = 3e-6
ADAM_LR = 0.0001
BATCH_SIZE = 64
DIM_C = 3
DIM_X = 32
DIM_Y = 32
DIM_Z = 128
EPOCHS = 35
ITERS_D = 1
ITERS_M = 1
NONLIN = tanh

GAUSSIAN_MEAN = 0.

#################################### MODELS ###################################

def discriminator(input_var=None, use_batch_norm=True, leak=0.02):
    if not use_batch_norm:
        bn = lambda x: x
    else:
        bn = batch_norm
    lrelu = LeakyRectify(leak)
    layer1 = InputLayer(shape=(None, DIM_C, DIM_X, DIM_Y), input_var=input_var)
    layer1 = Conv2DLayer(layer1, DIM_H, 4, stride=2, pad=1, nonlinearity=lrelu)
    layer2 = Conv2DLayer(layer1, DIM_H*2, 4, stride=2, pad=1,nonlinearity=lrelu)
    layer3 = bn(layer2)
    layer4 = Conv2DLayer(layer3, DIM_H*4, 4, stride=2, pad=1,nonlinearity=lrelu)
    layer5 = bn(layer4)
    layer6 = DenseLayer(layer5, 1, nonlinearity=None)
    return layer1, layer2, layer3, layer4, layer5, layer6

def generator(input_var=None, dim_z=DIM_Z, use_batch_norm=True):
    if not use_batch_norm:
        bn = lambda x: x
    else:
        bn = batch_norm
    layer1 = InputLayer(shape=(None, dim_z), input_var=input_var)
    layer1 = DenseLayer(layer1, DIM_H*4*4*4)
    layer2 = bn(layer1)
    layer3 = ReshapeLayer(layer2, ([0], DIM_H*4, 4, 4))
    layer3 = TransposedConv2DLayer(layer3, DIM_H*2, 4, stride=2, crop=1)
    layer4 = bn(layer3)
    layer5 = TransposedConv2DLayer(layer4, DIM_H, 4, stride=2, crop=1)
    layer6 = bn(layer5)
    layer7 = TransposedConv2DLayer(layer6, DIM_C, 4, stride=2, crop=1, nonlinearity=NONLIN)
    return layer1, layer2, layer3, layer4, layer5, layer6, layer7

###############################################################################

if __name__ == '__main__':
    if BN_D and BN_G:
        out_dir = '/u/grewalka/cifar10/%d/variance_lipschitz/%d_1/' % (DIM_H, ITERS_M)
    else:
        exit(0)    

    X, z = T.tensor4(), T.fmatrix()
    z = T.fmatrix()

    a1, a2, a3, a4, a5, D = discriminator(X, use_batch_norm=BN_D)
    G1, G2, G3, G4, G5, G6, G = generator(z, use_batch_norm=BN_G)

    y_real, D5, D4, D3, D2, D1 = get_output([D, a5, a4, a3, a2, a1])
    X_fake, G6, G5, G4, G3, G2, G1 = get_output([G, G6, G5, G4, G3, G2, G1])
    y_fake, D5_fake, D4_fake, D3_fake, D2_fake, D1_fake = get_output([D, a5, a4, a3, a2, a1], X_fake)

    actor_term = (y_real - y_fake) / ((X - X_fake)**2).sum(axis=(1,2,3)).sqrt()

    # Meta-discriminator
    M = real_fake_discriminator(actor_term)

    # Samples from Gaussian and output of meta-discriminator
    y_0 = T.fmatrix()
    m_real = get_output(M, y_0)
    m_fake = get_output(M)

    # Loss functions
    M_loss = (T.nnet.softplus(-m_real) + T.nnet.softplus(-m_fake) + m_fake).mean()
    D_loss = (m_fake**2).mean() - y_real.mean() + y_fake.mean()
    G_loss = y_real.mean() - y_fake.mean()

    # Updates to be performed during training
    updates_M = adam(loss_or_grads=M_loss,params=get_all_params(M, trainable=True),
        learning_rate=ADAM_LR,beta1=ADAM_BETA1,beta2=ADAM_BETA2,epsilon=ADAM_EPSILON)

    updates_D = adam(loss_or_grads=D_loss,params=get_all_params(D, trainable=True),
        learning_rate=ADAM_LR,beta1=ADAM_BETA1,beta2=ADAM_BETA2,epsilon=ADAM_EPSILON)

    updates_G = adam(loss_or_grads=G_loss,params=get_all_params(G, trainable=True),
        learning_rate=ADAM_LR,beta1=ADAM_BETA1,beta2=ADAM_BETA2,epsilon=ADAM_EPSILON)

    train_M = function([y_0, X, z], outputs=M_loss, updates=updates_M, allow_input_downcast=True)
    train_D = function([X, z], outputs=D_loss, updates=updates_D, allow_input_downcast=True)
    train_G = function([X, z], outputs=G_loss, updates=updates_G, allow_input_downcast=True)

    # Gradient Norms
    G_loss_grad_norm_val = function([X, z],outputs=(T.grad(G_loss, X_fake)**2).sum(axis=(1,2,3)).mean())
    D_loss_grad_norm_val = function([X, z],outputs=(T.grad(D_loss, X_fake)**2).sum(axis=(1,2,3)).mean())
    
    # Sampling functions
    z_samples = np.float32(np.random.normal(size=(BATCH_SIZE, DIM_Z)))
    generate = function([z], outputs=X_fake)
    D_out_real = function([X], outputs=y_real)
    D_out_fake = function([z], outputs=y_fake)

    # Load data
    stream, num_examples = get_data(BATCH_SIZE)
    print('Output files will be placed in: {}'.format(out_dir))

    for epoch in range(EPOCHS):
        print('Starting Epoch {}/{} ...'.format(epoch+1, EPOCHS))
        
        # Gradient norms
        G_loss_grad_norms = np.zeros(shape=(num_examples / BATCH_SIZE))
        D_loss_grad_norms = np.zeros(shape=(num_examples / BATCH_SIZE))
        D_samples = np.zeros(shape=((num_examples / BATCH_SIZE)*BATCH_SIZE, 2))
        
        # Training
        for i in range(num_examples / BATCH_SIZE):
            
            iterator = stream.get_epoch_iterator()
            for k in range(ITERS_D):
            
                # Train fake data discriminator
                y_0_i = np.float32(np.random.normal(loc=GAUSSIAN_MEAN,size=(BATCH_SIZE,1)))
                z_i = np.float32(np.random.normal(size=(BATCH_SIZE,DIM_Z)))
                x_i = iterator.next()[0]
                train_M(y_0_i, x_i, z_i)
                
                # Train discriminator
                z_i = np.float32(np.random.normal(size=(BATCH_SIZE,DIM_Z)))
                train_D(x_i, z_i)
                
            # train generator
            z_i = np.float32(np.random.normal(size=(BATCH_SIZE,DIM_Z)))
            train_G(x_i, z_i)
            
            # Get gradient norms
            if epoch < 15:
                iterator = stream.get_epoch_iterator()
                x_i = iterator.next()[0]
                z_i = np.float32(np.random.normal(size=(BATCH_SIZE, DIM_Z)))

                D_loss_grad_norms[i] = D_loss_grad_norm_val(x_i, z_i)
                G_loss_grad_norms[i] = G_loss_grad_norm_val(x_i, z_i)
        
        # Generate samples from G
        x_samples = generate(z_samples)
        with open(os.path.join(out_dir, 'x_samples_%d.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, x_samples)

        # Sample from D
        iterator = stream.get_epoch_iterator()
        for i in range(num_examples / BATCH_SIZE):
            x_i = iterator.next()[0]
            z_i = np.float32(np.random.normal(size=(BATCH_SIZE,DIM_Z)))
            D_samples[i*BATCH_SIZE:(i+1)*BATCH_SIZE,0:1] = D_out_real(x_i)
            D_samples[i*BATCH_SIZE:(i+1)*BATCH_SIZE,1:2] = D_out_fake(z_i)

        if epoch < 15:
            with open(os.path.join(out_dir, 'D_loss_grad_norms_%d.npz' % (epoch+1)), 'w+') as f:
                np.savez(f, D_loss_grad_norms)
            with open(os.path.join(out_dir, 'G_loss_grad_norms_%d.npz' % (epoch+1)), 'w+') as f:
                np.savez(f, G_loss_grad_norms)

            D_loss_grad_norms, G_loss_grad_norms = None, None

        if epoch < 10:
            D_params = get_all_params(get_all_layers(D))
            with open(os.path.join(out_dir, 'discriminator_model_{}.npz'.format(epoch+1)), 'w+') as f:
                np.savez(f, D_params)
            G_params = get_all_params(get_all_layers(G))
            with open(os.path.join(out_dir, 'generator_model_{}.npz'.format(epoch+1)), 'w+') as f:
                np.savez(f, G_params)
        
        # Save Results
        with open(os.path.join(out_dir, 'D_samples_%d.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, D_samples)

