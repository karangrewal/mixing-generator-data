#!/usr/bin/env python

"""
Implementation of WGAN-GP. (https://arxiv.org/abs/1704.00028)
"""

import numpy as np
from lasagne.layers import get_all_params, get_output, get_all_layers
from lasagne.updates import adam
from theano import function
import theano.tensor as T
import os

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
        'epochs':40,
        'lambda':10
    }

    out_dir = '/u/grewalka/lasagne/wasserstein-gp/%d_1_bn/' % (params['discriminator_iters'])

    with open(os.path.join(out_dir, 'out.log'), 'w+') as f:
        f.write('WGAN-GP\n')
        f.write('D iters: {}'.format(params['discriminator_iters']))

    X = T.tensor4()
    z = T.fmatrix()
    epsilon = T.fscalar()

    D, G = discriminator(X, use_batch_norm=True), generator(z, use_batch_norm=True)

    y_real = get_output(D)
    X_fake = get_output(G)
    y_fake = get_output(D, X_fake)
    y_inter = get_output(D, epsilon * X + (1.-epsilon) * X_fake)

    # Sampling functions
    generate = function([z], outputs=X_fake)
    D_out_R = function([X], outputs=y_real)
    D_out_F = function([z], outputs=y_fake)

    # Gradient Penalty; might be scaled by factor `epsilon`
    penalty = np.sqrt((T.grad(y_inter.sum(), X)**2).sum(axis=(1,2,3))) # Norm
    penalty = (penalty - 1) ** 2

    # WGAN-GP loss
    D_loss = (y_fake - y_real).mean() + params['lambda'] * penalty.mean()
    G_loss = (-y_fake).mean()

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

    train_D = function([X, z, epsilon],outputs=D_loss,updates=updates_D,allow_input_downcast=True)
    train_G = function([z],outputs=G_loss,updates=updates_G,allow_input_downcast=True)

    # Gradient Norms
    # D_grad_real = T.grad(y_real.mean(), X)
    # D_grad_fake = T.grad(y_fake.mean(), X_fake)
    
    # # Value of E||grad(dL/dx)||^2
    # D_grad_real_norm_value = function([X],outputs=(D_grad_real**2).sum(axis=(1,2,3)).mean())
    # D_grad_fake_norm_value = function([z],outputs=(D_grad_fake**2).sum(axis=(1,2,3)).mean())
    
    # Load data
    stream, num_examples = get_data(params['batch_size'])

    print('D_iters: {}'.format(params['discriminator_iters']))
    qr = 0
    for epoch in range(params['epochs']):
        print('Starting Epoch {}/{} ...'.format(epoch+1, params['epochs']))
        
        # Keep track of losses and gradient norms
        D_losses = np.zeros(shape=(num_examples / params['batch_size'], params['discriminator_iters']))
        G_losses = np.zeros(shape=(num_examples / params['batch_size']))
        D_grad_real_norms = np.zeros(shape=(num_examples / params['batch_size'], params['discriminator_iters']))
        D_grad_fake_norms = np.zeros(shape=(num_examples / params['batch_size'], params['discriminator_iters']))

        # D samples
        D_samples = np.zeros(shape=(num_examples, 2))
        D_samples.fill(99.)

        for i in range(num_examples / params['batch_size']):

            # Train Discriminator
            iterator = stream.get_epoch_iterator()
            for k in range(params['discriminator_iters']):
                z_i = np.float32(np.random.normal(size=(params['batch_size'],100)))# CHANGE TO DIMZ
                x_i = iterator.next()[0]
                qr += 1                 
                with open(os.path.join(out_dir, 'x_%d.npz' % qr), 'w+') as f:
                    np.savez(f, x_i)
                epsilon_i = np.random.uniform()
                
                D_losses[i,k] = train_D(x_i, z_i, epsilon_i)
                # D_grad_real_norms[i,k] = D_grad_real_norm_value(x_i)
                # D_grad_fake_norms[i,k] = D_grad_fake_norm_value(z_i)

            # train generator
            z_i = np.float32(np.random.normal(size=(params['batch_size'],100)))# CHANGE TO DIMZ
            G_losses[i] = train_G(z_i)

        # Generate samples from G
        z_i = np.float32(np.random.normal(size=(params['batch_size'],100)))
        x_samples = generate(z_i)
        with open(os.path.join(out_dir, 'x_samples_%d.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, x_samples, delimiter=',')

        # Sample from D
        iterator = stream.get_epoch_iterator()
        for i in range(num_examples / params['batch_size']):
            x_i = iterator.next()[0]
            z_i = np.float32(np.random.normal(size=(params['batch_size'],100)))
            D_samples[i*params['batch_size']:(i+1)*params['batch_size'],0:1] = D_out_R(x_i)
            D_samples[i*params['batch_size']:(i+1)*params['batch_size'],1:2] = D_out_F(z_i)

        # Save Results
        with open(os.path.join(out_dir, 'D_samples_%d.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, D_samples, delimiter=',')

        with open(os.path.join(out_dir, '%d_D_loss.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, D_losses, delimiter=',')
        with open(os.path.join(out_dir, '%d_G_loss.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, G_losses, delimiter=',')

        with open(os.path.join(out_dir, '%d_D_grad_real_norms.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, D_grad_real_norms, delimiter=',')
        with open(os.path.join(out_dir, '%d_D_grad_fake_norms.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, D_grad_fake_norms, delimiter=',')
        
        # Save model
        G_params = get_all_params(get_all_layers(G))
        with open(os.path.join(out_dir, 'generator_model_%d.npz' % params['epochs']), 'w+') as f:
            np.savez(f, G_params)
        
        with open(os.path.join(out_dir, 'weights_%d.txt' % (epoch+1)), 'w+') as f:
            p = get_all_params(D)
            f.write('min: {} max: {}\n'.format(np.min(p[0].get_value()), np.max(p[0].get_value())))
            f.write('min: {} max: {}\n'.format(np.min(p[2].get_value()), np.max(p[2].get_value())))
            f.write('min: {} max: {}\n'.format(np.min(p[4].get_value()), np.max(p[4].get_value())))
            f.write('min: {} max: {}\n'.format(np.min(p[6].get_value()), np.max(p[6].get_value())))
