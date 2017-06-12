#!/usr/bin/env python

"""
Implementation of Least Squares GAN. (https://arxiv.org/abs/1611.04076)
"""

import numpy as np
from lasagne.layers import get_all_params, get_output, get_all_layers
from lasagne.objectives import binary_crossentropy
from lasagne.updates import adam, sgd
from theano import function
import theano.tensor as T

from data import get_data
from model import discriminator, generator

if __name__ == '__main__':
    # place params in separate file
    params = {
        'adam_beta1':0.5,
        'adam_beta2':0.999,
        'adam_epsilon':3e-6,
        'adam_learning_rate':0.0001,
        'batch_size': 64,
        'discriminator_iters':1,
        'epochs':50,
    }

    X = T.tensor4()
    z = T.fmatrix()

    D, G = discriminator(X), generator(z)

    y_real = get_output(D)
    X_fake = get_output(G)
    y_fake = get_output(D, X_fake)
    
    # Least Squares loss
    D_loss = 0.5 * ((y_real - 1) ** 2).mean() + 0.5 * ((y_fake + 1) ** 2).mean()
    G_loss = 0.5 * ((y_fake - 1) ** 2).mean()

    updates_D = adam(
        loss_or_grads=D_loss,
        params=get_all_params(D, trainable=True),
        learning_rate=params['adam_learning_rate'],
        beta1=params['adam_beta1'],
        beta2=params['adam_beta2'],
        epsilon=params['adam_epsilon']
    )

    updates_G = adam(
        loss_or_grads=G_loss,
        params=get_all_params(G, trainable=True),
        learning_rate=params['adam_learning_rate'],
        beta1=params['adam_beta1'],
        beta2=params['adam_beta2'],
        epsilon=params['adam_epsilon']
    )

    train_D = function(
        [X, z],
        outputs=D_loss,
        updates=updates_D,
        allow_input_downcast=True
    )

    train_G = function(
        [z],
        outputs=G_loss,
        updates=updates_G,
        allow_input_downcast=True
    )

    # Gradient Norms
    D_grad_norm = (T.grad(D_loss, X) ** 2).sum(axis=(0,1,2,3))
    G_grad_norm = (T.grad(G_loss, X_fake) ** 2).sum(axis=(0,1,2,3))

    # Value of E||grad(dL/dx)||^2
    D_grad_norm_value = function([X, z],outputs=D_grad_norm)
    G_grad_norm_value = function([z],outputs=G_grad_norm)

    # Load data
    batches = get_data(params['batch_size'])

    # Sampling functions
    generate = function([z], outputs=X_fake)
    D_out_R = function([X], outputs=y_real)
    D_out_F = function([z], outputs=y_fake)

    for epoch in range(params['epochs']):
        print('\nStarting Epoch {}/{} ...\n'.format(epoch+1, params['epochs']))
        
        # Keep track of losses and gradient norms
        D_losses = np.zeros(shape=(batches.shape[0], params['discriminator_iters']))
        G_losses = np.zeros(shape=(batches.shape[0]))
        D_grad_norms = np.zeros(shape=(batches.shape[0], params['discriminator_iters']))
        # G_grad_norms = np.zeros(shape=(batches.shape[0]))

        # D samples
        D_samples = np.zeros(shape=(batches.shape[0]*32, 2))
        D_samples.fill(99.)

        for i in range(batches.shape[0]):

            # Train Discriminator
            for k in range(params['discriminator_iters']):
                x_i = batches[i]
                z_i = np.float32(np.random.normal(size=(params['batch_size'],100)))# CHANGE TO DIMZ
                D_losses[i,k] = train_D(x_i, z_i)
                D_grad_norms[i,k] = D_grad_norm_value(x_i, z_i)

            # train generator
            z_i = np.float32(np.random.normal(size=(params['batch_size'],100)))# CHANGE TO DIMZ
            G_losses[i] = train_G(z_i)
            # G_grad_norms[i] = G_grad_norm_value(z_i)

        # Generate samples from G
        z_i = np.float32(np.random.normal(size=(params['batch_size'],100)))
        x_samples = generate(z_i)
        with open('/u/grewalka/lasagne/least_squares/x_samples_%d.npz' % (epoch+1), 'w+') as f:
            np.savez(f, x_samples, delimiter=',')

        # Sample from D
        for i in range(batches.shape[0]):
            x_i = batches[i]
            z_i = np.float32(np.random.normal(size=(params['batch_size'],100)))
            D_samples[i*32:i*32+params['batch_size'],0:1] = D_out_R(x_i)
            D_samples[i*32:i*32+params['batch_size'],1:2] = D_out_F(z_i)

        # End of epoch
        with open('/u/grewalka/lasagne/least_squares/D_grad_norms_%d.npz' % (epoch+1), 'w+') as f:
            np.savez(f, D_grad_norms, delimiter=',')

        with open('/u/grewalka/lasagne/least_squares/D_samples_%d.npz' % (epoch+1), 'w+') as f:
            np.savez(f, D_samples, delimiter=',')
