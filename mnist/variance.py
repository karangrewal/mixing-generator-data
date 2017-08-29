#!/usr/bin/env python

"""
Increasing variance in the discriminator output of GANs by forcing it's 
marginal distributions D(X|x~p) and D(X|x~q) to be gaussian with means 1 and 0 
respectively.
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

if __name__ == '__main__':
    print('starting')
    params = {
        'adam_beta1':0.5,
        'adam_beta2':0.999,
        'adam_epsilon':3e-6,
        'adam_learning_rate':0.0001,
        'batch_size':64,
        'D_loss': 'proxy',
        'dim_z':100,
        'iters_D':1,
        'iters_F':1,
        'iters_R':1,
        'epochs':35
    }
    
    out_dir = '/u/grewalka/lasagne/variance/%d_%d_%d_1/' % (params['iters_F'], params['iters_R'], params['iters_D'])
    
    with open(os.path.join(out_dir, 'out.log'), 'w+') as f:
        f.write('Variance-maximizing GAN\n')
        f.write('F iters: {}, R iters: {}, D iters: {}'.format(params['iters_F'], params['iters_R'], params['iters_D']))

    X = T.tensor4()
    z = T.fmatrix()

    D, G = discriminator(X, use_batch_norm=True), generator(z, use_batch_norm=True)

    y_real = get_output(D)
    X_fake = get_output(G)
    y_fake = get_output(D, X_fake)

    # Real and fake discriminators
    F, R = real_fake_discriminator(y_fake), real_fake_discriminator(y_real)

    # Samples from N(0,1) and N(1,1)
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
    if params['D_loss'] == 'regular':
        D_loss = (-(T.nnet.softplus(-f_fake) + T.nnet.softplus(-r_fake) + f_fake + r_fake)).mean()
    elif params['D_loss'] == 'proxy':
        D_loss = (T.nnet.softplus(-f_fake) + T.nnet.softplus(-r_fake)).mean()
    G_loss = 0.5 * ((y_fake - 1) ** 2).mean()

    # Updates to be performed during training
    updates_F = adam(
        loss_or_grads=F_loss,
        params=get_all_params(F, trainable=True),
        learning_rate=params['adam_learning_rate'],
        beta1=params['adam_beta1'],
        beta2=params['adam_beta2'],
        epsilon=params['adam_epsilon']
    )

    updates_R = adam(
        loss_or_grads=R_loss,
        params=get_all_params(R, trainable=True),
        learning_rate=params['adam_learning_rate'],
        beta1=params['adam_beta1'],
        beta2=params['adam_beta2'],
        epsilon=params['adam_epsilon']
    )

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

    train_F = function([v_0, z], outputs=F_loss, updates=updates_F, allow_input_downcast=True)
    train_R = function([v_1, X], outputs=R_loss, updates=updates_R, allow_input_downcast=True)
    train_D = function([X, z], outputs=D_loss, updates=updates_D, allow_input_downcast=True)
    train_G = function([z], outputs=G_loss, updates=updates_G, allow_input_downcast=True)

    # Gradient Norms
    D_grad_real = T.grad(y_real.mean(), X)
    D_grad_fake = T.grad(y_fake.mean(), X_fake)
    G_loss_grad = T.grad(G_loss, X_fake)
    G_loss_grad_norm_value = function([z],outputs=(G_loss_grad**2).sum(axis=(1,2,3)).mean())
    D_loss_grad = T.grad(D_loss, X_fake)
    D_loss_grad_norm_value = function([X, z],outputs=(D_loss_grad**2).sum(axis=(1,2,3)).mean())

    # Value of E||dF/dx||^2, E||dR/dx||^2, E||dD/dx||^2
    D_grad_real_norm_value = function([X],outputs=(D_grad_real**2).sum(axis=(1,2,3)).mean())
    D_grad_fake_norm_value = function([z],outputs=(D_grad_fake**2).sum(axis=(1,2,3)).mean())
    
    # Sampling functions
    generate = function([z], outputs=X_fake)
    D_out_R = function([X], outputs=y_real)
    D_out_F = function([z], outputs=y_fake)
    
    # Load data
    stream, num_examples = get_data(params['batch_size'])

    print('VRAL with meta-discriminators, training ratio = {}:{}:{}:1'.format(params['iters_F'], params['iters_R'], params['iters_D']))
    print('Output files will be placed in: {}'.format(out_dir))

    for epoch in range(params['epochs']):
        print('Starting Epoch {}/{} ...'.format(epoch+1, params['epochs']))
        
        # Keep track of Information
        F_losses = np.zeros(shape=(num_examples / params['batch_size'], params['iters_F']))
        R_losses = np.zeros(shape=(num_examples / params['batch_size'], params['iters_R']))
        D_losses = np.zeros(shape=(num_examples / params['batch_size'], params['iters_D']))
        G_losses = np.zeros(shape=(num_examples / params['batch_size']))

        D_grad_real_norms = np.zeros(shape=(num_examples / params['batch_size'], params['iters_D']))
        D_grad_fake_norms = np.zeros(shape=(num_examples / params['batch_size'], params['iters_D']))
        
        G_loss_grad_norms = np.zeros(shape=(num_examples / params['batch_size']))
        D_loss_grad_norms = np.zeros(shape=(num_examples / params['batch_size'], params['iters_D']))
        
        D_samples = np.zeros(shape=((num_examples / params['batch_size'])*params['batch_size'] , 2))
        
        # Training
        for i in range(num_examples / params['batch_size']):
            
            iterator = stream.get_epoch_iterator()
            for k in range(params['iters_D']):
            
                # Train fake data discriminator
                v_0_i = np.float32(np.random.normal(size=(params['batch_size'],1)))
                z_i = np.float32(np.random.normal(size=(params['batch_size'],params['dim_z'])))
                train_F(v_0_i, z_i)
                
                # Train real data discriminator
                v_1_i = np.float32(np.random.normal(loc=1.0,size=(params['batch_size'],1)))
                x_i = iterator.next()[0]
                train_R(v_1_i, x_i)
                
                # Train discriminator
                z_i = np.float32(np.random.normal(size=(params['batch_size'],params['dim_z'])))
                train_D(x_i, z_i)
                D_grad_real_norms[i,k] = D_grad_real_norm_value(x_i)
                D_grad_fake_norms[i,k] = D_grad_fake_norm_value(z_i)
                D_loss_grad_norms[i,k] = D_loss_grad_norm_value(x_i, z_i)
            
            # train generator
            z_i = np.float32(np.random.normal(size=(params['batch_size'],params['dim_z'])))
            G_losses[i] = train_G(z_i)
            G_loss_grad_norms[i] = G_loss_grad_norm_value(z_i)

        # Generate samples from G
        z_i = np.float32(np.random.normal(size=(params['batch_size'],params['dim_z'])))
        x_samples = generate(z_i)
        with open(os.path.join(out_dir, 'x_samples_%d.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, x_samples)

        # Sample from D
        iterator = stream.get_epoch_iterator()
        for i in range(num_examples / params['batch_size']):
            x_i = iterator.next()[0]
            z_i = np.float32(np.random.normal(size=(params['batch_size'],params['dim_z'])))
            D_samples[i*params['batch_size']:(i+1)*params['batch_size'],0:1] = D_out_R(x_i)
            D_samples[i*params['batch_size']:(i+1)*params['batch_size'],1:2] = D_out_F(z_i)

        # Save Results
        with open(os.path.join(out_dir, 'D_samples_%d.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, D_samples)
        
        with open(os.path.join(out_dir, '%d_D_loss.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, D_losses)
        with open(os.path.join(out_dir, '%d_G_loss.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, G_losses)

        with open(os.path.join(out_dir, '%d_D_grad_real_norms.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, D_grad_real_norms)
        with open(os.path.join(out_dir, '%d_D_grad_fake_norms.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, D_grad_fake_norms)
        
        with open(os.path.join(out_dir, 'D_loss_grad_norms_%d.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, D_loss_grad_norms)
        with open(os.path.join(out_dir, 'G_loss_grad_norms_%d.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, G_loss_grad_norms)
