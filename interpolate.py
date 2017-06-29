#!/usr/bin/env python

"""
Gamma Experiment:
    Interpolate between x and g(z) and measure the output of the discriminator
"""

import numpy as np
from lasagne.layers import get_all_params, get_output, get_all_layers, set_all_param_values
from lasagne.updates import adam
from theano import function
import theano.tensor as T
import os

from data import get_data
from model import discriminator, generator
from real_fake import real_fake_discriminator

if __name__ == '__main__':
    # place params in separate file
    params = {
        'adam_beta1':0.5,
        'adam_beta2':0.999,
        'adam_epsilon':3e-6,
        'adam_learning_rate':0.0001,
        'batch_size':64,
        'dim_z':100,
        'iters_D':1,
        'iters_F':1,
        'iters_R':1,
        'epochs':20,
        'load_model':False
    }

    out_dir = '/u/grewalka/lasagne/gamma-experiment/variance/%d_%d_%d_1/' % (params['iters_F'], params['iters_R'], params['iters_D'])
    
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
    D_grad = T.grad(y_real.mean(), X)
    D_grad_norm_value = function([X],outputs=(D_grad**2).sum(axis=(1,2,3)).mean())

    # Load data
    stream, num_examples = get_data(params['batch_size'])

    # Sampling functions
    generate = function([z], outputs=X_fake)
    D_out = function([X], outputs=y_real)
    
    D_grad_norms = np.zeros(shape=(100, 32))
    D_samples = np.zeros(shape=(100, params['batch_size'], 32))
    
    if params['load_model']:
        
        # Load parameters
        with open(os.path.join(out_dir, 'discriminator_model.npz')) as f:
            D_params = np.load(f)['arr_0']
        set_all_param_values(D, [param.get_value() for param in D_params])
        
        with open(os.path.join(out_dir, 'generator_model.npz')) as f:
            G_params = np.load(f)['arr_0']
        set_all_param_values(G, [param.get_value() for param in G_params])
        
        k = 0
        for gamma in (np.array(range(0, 100)) / 100.):
    
            for n in range(32):
                z_i = np.float32(np.random.normal(size=(params['batch_size'],params['dim_z'])))
                x_fake = generate(z_i)
                iterator = stream.get_epoch_iterator()
                x_real = iterator.next()[0]
                
                x_hat = np.float32(gamma * x_fake + (1.-gamma) * x_real)
                D_samples[k,:,n] = D_out(x_hat).reshape(params['batch_size'])
                D_grad_norms[k,n] = D_grad_norm_value(x_hat)
            k += 1
            
        with open(os.path.join(out_dir, 'D_samples_gamma_{}.npz'.format(params['epochs'])), 'w+') as f:
            np.savez(f, D_samples)
            print('Wrote to: D_samples_gamma_{}.npz'.format(params['epochs']))
        with open(os.path.join(out_dir, 'D_grad_norms_gamma_{}.npz'.format(params['epochs'])), 'w+') as f:
            np.savez(f, D_grad_norms)
            print('Wrote to: D_grad_norms_gamma_{}.npz'.format(params['epochs']))
    
    else:
        
        with open(os.path.join(out_dir, 'out.log'), 'w+') as f:
            f.write('gamma samples after {} epochs'.format(params['epochs']))
            f.write('{}:{}:{}:1'.format(params['iters_F'], params['iters_R'], params['iters_D']))

        print('F iters: {}, R iters: {}, D iters: {}'.format(params['iters_F'], params['iters_R'], params['iters_D']))
        print('Output files will be placed in: {}'.format(out_dir))
    
        for epoch in range(params['epochs']):
            print('\nStarting Epoch {}/{} ...\n'.format(epoch+1, params['epochs']))
    
            # Training
            for i in range(num_examples / params['batch_size']):
    
                 # Train fake data discriminator
                iterator = stream.get_epoch_iterator()
                for k in range(params['iters_F']):
    
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
                    
                # train generator
                z_i = np.float32(np.random.normal(size=(params['batch_size'],params['dim_z'])))
                train_G(z_i)
    
            D_grad_norms.fill(0.)
            D_samples.fill(0.)
    
            # **
            # Interpolate
            # **
            k = 0
            for gamma in (np.array(range(0, 100)) / 100.):
        
                for n in range(32):
                    z_i = np.float32(np.random.normal(size=(params['batch_size'],params['dim_z'])))
                    x_fake = generate(z_i)
                    iterator = stream.get_epoch_iterator()
                    x_real = iterator.next()[0]
                    
                    x_hat = np.float32(gamma * x_fake + (1.-gamma) * x_real)
                    D_samples[k,:, n] = D_out(x_hat).reshape(params['batch_size'])
                    D_grad_norms[k, n] = D_grad_norm_value(x_hat)
                k += 1
                
            with open(os.path.join(out_dir, 'D_samples_gamma_{}.npz'.format(epoch+1)), 'w+') as f:
                np.savez(f, D_samples)
            with open(os.path.join(out_dir, 'D_grad_norms_gamma_{}.npz'.format(epoch+1)), 'w+') as f:
                np.savez(f, D_grad_norms)
        
        # Save model
        D_params = get_all_params(get_all_layers(D))
        with open(os.path.join(out_dir, 'discriminator_model.npz'), 'w+') as f:
            np.savez(f, D_params)
        
        G_params = get_all_params(get_all_layers(G))
        with open(os.path.join(out_dir, 'generator_model.npz'), 'w+') as f:
            np.savez(f, G_params)
