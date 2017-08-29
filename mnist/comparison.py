#!/usr/bin/env python

"""
Performs the following:
    1. Train a generator from scratch on MNIST
    2. Holding the generator fixed, train discriminators 
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
    params = {
        'adam_beta1':0.5,
        'adam_beta2':0.999,
        'adam_epsilon':3e-6,
        'adam_learning_rate':0.0001,
        'batch_size':64,
        'dim_z':100,
        'iters_G':5,
        'epochs':60,
        'load_generator':False
    }

    out_dir = '/u/grewalka/lasagne/comparison/%d/' % params['iters_G']

    ###########################################################################
    ########################### [1] Train Generator ###########################
    ###########################################################################

    X, z = T.tensor4(), T.fmatrix()
    G = generator(z, use_batch_norm=True)
    
    if params['load_generator']:
        with open(os.path.join(out_dir, 'generator_model.npz')) as f:
            G_params = np.load(f)['arr_0']
        set_all_param_values(G, [param.get_value() for param in G_params])
    else:
        D = discriminator(X, use_batch_norm=True)
        y_real, X_fake = get_output(D), get_output(G)
        y_fake = get_output(D, X_fake)
    
        # LSE Loss
        D_loss = 0.5 * ((y_real - 1) ** 2).mean() + 0.5 * ((y_fake + 1) ** 2).mean()
        G_loss = 0.5 * ((y_fake - 1) ** 2).mean()
    
        # Updates to be performed during training
        updates_D = adam(loss_or_grads=D_loss,params=get_all_params(D, trainable=True),
            learning_rate=params['adam_learning_rate'],beta1=params['adam_beta1'],
            beta2=params['adam_beta2'],epsilon=params['adam_epsilon'])
    
        updates_G = adam(loss_or_grads=G_loss,params=get_all_params(G, trainable=True),
            learning_rate=params['adam_learning_rate'],beta1=params['adam_beta1'],
            beta2=params['adam_beta2'],epsilon=params['adam_epsilon'])
        
        train_D = function([X, z], outputs=D_loss, updates=updates_D, allow_input_downcast=True)
        train_G = function([z], outputs=G_loss, updates=updates_G, allow_input_downcast=True)
    
        # Sampling functions
        generate = function([z], outputs=X_fake)

        # Load data
        stream, num_examples = get_data(params['batch_size'])
    
        print('Output files will be placed in: {}'.format(out_dir))
        for epoch in range(5):
            print('\nTraining Generator. Epoch {}/20 ...\n'.format(epoch+1))
    
            # Training
            for i in range(num_examples / params['batch_size']):
    
                # traing discriminator
                iterator = stream.get_epoch_iterator()
                x_i = iterator.next()[0]
                z_i = np.float32(np.random.normal(size=(params['batch_size'],params['dim_z'])))
                train_D(x_i, z_i)
                
                # train generator
                z_i = np.float32(np.random.normal(size=(params['batch_size'],params['dim_z'])))
                train_G(z_i)
    
            # Generate samples from G
            z_i = np.float32(np.random.normal(size=(params['batch_size'],params['dim_z'])))
            x_samples = generate(z_i)
            with open(os.path.join(out_dir, 'generated_samples.npz'), 'w+') as f:
                np.savez(f, x_samples)
                
        # Save Generator
        G_params = get_all_params(get_all_layers(G))
        with open(os.path.join(out_dir, 'generator_model.npz'), 'w+') as f:
            np.savez(f, G_params)

    ###########################################################################
    ############## [2] Train Discriminators, Hold Generator Fixed #############
    ###########################################################################

    X_regular, z_regular = T.tensor4(), T.fmatrix()
    X_lsgan, z_lsgan = T.tensor4(), T.fmatrix()
    X_wgan, z_wgan = T.tensor4(), T.fmatrix()
    X_vral, z_vral = T.tensor4(), T.fmatrix()
    
    # Discriminators
    D_regular, D_lsgan, D_wgan, D_vral = discriminator(X_regular, use_batch_norm=True), discriminator(X_lsgan, use_batch_norm=True), discriminator(X_wgan, use_batch_norm=True), discriminator(X_vral, use_batch_norm=True)
    
    y_real_regular, X_fake_regular = get_output(D_regular), get_output(G)
    y_real_lsgan, X_fake_lsgan = get_output(D_lsgan), get_output(G)
    y_real_wgan, X_fake_wgan = get_output(D_wgan), get_output(G)
    y_real_vral, X_fake_vral = get_output(D_vral), get_output(G)
    
    y_fake_regular = get_output(D_regular, X_fake_regular)
    y_fake_lsgan = get_output(D_lsgan, X_fake_lsgan)
    y_fake_wgan = get_output(D_wgan, X_fake_wgan)
    y_fake_vral = get_output(D_vral, X_fake_vral)
    
    F, R = real_fake_discriminator(y_fake_vral), real_fake_discriminator(y_real_vral)

    # Samples from N(0,1) and N(1,1)
    y_0, y_1 = T.fmatrix(), T.fmatrix()

    # Outputs of real and fake discriminators
    r_real, r_fake = get_output(R, y_1), get_output(R)
    f_real, f_fake = get_output(F, y_0), get_output(F)

    # Loss Functions
    D_loss_regular = (y_fake_regular + T.nnet.softplus(-y_real_regular) + T.nnet.softplus(-y_fake_regular)).mean()
    D_loss_lsgan = 0.5 * ((y_real_lsgan - 1) ** 2).mean() + 0.5 * ((y_fake_lsgan + 1) ** 2).mean()
    D_loss_wgan = (y_fake_wgan - y_real_wgan).mean()
    
    D_loss_vral = (T.nnet.softplus(-f_fake) + T.nnet.softplus(-r_fake)).mean()
    F_loss = (T.nnet.softplus(-f_real) + T.nnet.softplus(-f_fake) + f_fake).mean()
    R_loss = (T.nnet.softplus(-r_real) + T.nnet.softplus(-r_fake) + r_fake).mean()

    # updates
    updates_D_regular = adam(loss_or_grads=D_loss_regular,
        params=get_all_params(D_regular, trainable=True),
        learning_rate=params['adam_learning_rate'],beta1=params['adam_beta1'],
        beta2=params['adam_beta2'],epsilon=params['adam_epsilon'])
    updates_D_lsgan = adam(loss_or_grads=D_loss_lsgan,
        params=get_all_params(D_lsgan, trainable=True),
        learning_rate=params['adam_learning_rate'],beta1=params['adam_beta1'],
        beta2=params['adam_beta2'],epsilon=params['adam_epsilon'])
    updates_D_wgan = adam(loss_or_grads=D_loss_wgan,
        params=get_all_params(D_wgan, trainable=True),
        learning_rate=params['adam_learning_rate'],beta1=params['adam_beta1'],
        beta2=params['adam_beta2'],epsilon=params['adam_epsilon'])
    
    updates_D_vral = adam(loss_or_grads=D_loss_vral,
        params=get_all_params(D_vral, trainable=True),
        learning_rate=params['adam_learning_rate'],beta1=params['adam_beta1'],
        beta2=params['adam_beta2'],epsilon=params['adam_epsilon'])
    updates_F = adam(loss_or_grads=F_loss,params=get_all_params(F, trainable=True),
        learning_rate=params['adam_learning_rate'],beta1=params['adam_beta1'],
        beta2=params['adam_beta2'],epsilon=params['adam_epsilon'])
    updates_R = adam(loss_or_grads=R_loss,params=get_all_params(R, trainable=True),
        learning_rate=params['adam_learning_rate'],beta1=params['adam_beta1'],
        beta2=params['adam_beta2'],epsilon=params['adam_epsilon'])


    train_D_regular = function([X_regular, z], outputs=D_loss_regular,
        updates=updates_D_regular, allow_input_downcast=True)
    train_D_lsgan = function([X_lsgan, z], outputs=D_loss_lsgan,
        updates=updates_D_lsgan, allow_input_downcast=True)
    train_D_wgan = function([X_wgan, z], outputs=D_loss_wgan,
        updates=updates_D_wgan, allow_input_downcast=True)

    train_D_vral = function([X_vral, z], outputs=D_loss_vral,
        updates=updates_D_vral, allow_input_downcast=True)
    train_F = function([y_0, z], outputs=F_loss, updates=updates_F,
        allow_input_downcast=True)
    train_R = function([y_1, X_vral], outputs=R_loss, updates=updates_R,
        allow_input_downcast=True)

    # Gradient Norms
    D_grad_fake_regular = T.grad(y_fake_regular.mean(), X_fake_regular)
    D_grad_fake_lsgan = T.grad(y_fake_lsgan.mean(), X_fake_lsgan)
    D_grad_fake_wgan = T.grad(y_fake_wgan.mean(), X_fake_wgan)
    D_grad_fake_vral = T.grad(y_fake_vral.mean(), X_fake_vral)

    # Gradient Norm Functions
    D_grad_fake_regular_norm_value = function([z],outputs=(D_grad_fake_regular**2).sum(axis=(1,2,3)).mean())
    D_grad_fake_lsgan_norm_value = function([z],outputs=(D_grad_fake_lsgan**2).sum(axis=(1,2,3)).mean())
    D_grad_fake_wgan_norm_value = function([z],outputs=(D_grad_fake_wgan**2).sum(axis=(1,2,3)).mean())
    D_grad_fake_vral_norm_value = function([z],outputs=(D_grad_fake_vral**2).sum(axis=(1,2,3)).mean())

    # Sampling functions
    D_regular_out_R = function([X_regular], outputs=y_real_regular)
    D_regular_out_F = function([z], outputs=y_fake_regular)

    D_lsgan_out_R = function([X_lsgan], outputs=y_real_lsgan)
    D_lsgan_out_F = function([z], outputs=y_fake_lsgan)

    D_wgan_out_R = function([X_wgan], outputs=y_real_wgan)
    D_wgan_out_F = function([z], outputs=y_fake_wgan)

    D_vral_out_R = function([X_vral], outputs=y_real_vral)
    D_vral_out_F = function([z], outputs=y_fake_vral)

    # Load data
    stream, num_examples = get_data(params['batch_size'])

    for epoch in range(params['epochs']):
        print('Starting Epoch {}/{} ...'.format(epoch+1, params['epochs']))

        # Keep track of Information
        # Losses
        D_losses_1 = np.zeros(shape=(num_examples / params['batch_size']))
        D_losses_2 = np.zeros(shape=(num_examples / params['batch_size']))
        D_losses_3 = np.zeros(shape=(num_examples / params['batch_size']))
        D_losses_4 = np.zeros(shape=(num_examples / params['batch_size']))

        # Gradient Norms
        D_grad_norms_1 = np.zeros(shape=(num_examples / params['batch_size']))
        D_grad_norms_2 = np.zeros(shape=(num_examples / params['batch_size']))
        D_grad_norms_3 = np.zeros(shape=(num_examples / params['batch_size']))
        D_grad_norms_4 = np.zeros(shape=(num_examples / params['batch_size']))

        # Output Distributions
        D_samples_regular = np.zeros(shape=((num_examples / params['batch_size'])*params['batch_size'], 2))
        D_samples_lsgan = np.zeros(shape=((num_examples / params['batch_size'])*params['batch_size'], 2))
        D_samples_wgan = np.zeros(shape=((num_examples / params['batch_size'])*params['batch_size'], 2))
        D_samples_vral = np.zeros(shape=((num_examples / params['batch_size'])*params['batch_size'], 2))

        # Training
        for i in range(num_examples / params['batch_size']):

            iterator = stream.get_epoch_iterator()

            # Train Regular Discriminator
            x_i = iterator.next()[0]
            z_i = np.float32(np.random.normal(size=(params['batch_size'],params['dim_z'])))
            D_losses_1[i] = train_D_regular(x_i, z_i)
            D_grad_norms_1[i] = D_grad_fake_regular_norm_value(z_i)

            # Train Least Squares Discriminator
            x_i = iterator.next()[0]
            z_i = np.float32(np.random.normal(size=(params['batch_size'],params['dim_z'])))
            D_losses_2[i] = train_D_lsgan(x_i, z_i)
            D_grad_norms_2[i] = D_grad_fake_lsgan_norm_value(z_i)

            # Train Regular Discriminator
            x_i = iterator.next()[0]
            z_i = np.float32(np.random.normal(size=(params['batch_size'],params['dim_z'])))
            D_losses_3[i] = train_D_wgan(x_i, z_i)
            D_grad_norms_3[i] = D_grad_fake_wgan_norm_value(z_i)

            # Train fake (meta) discriminator
            y_0_i = np.float32(np.random.normal(size=(params['batch_size'],1)))
            z_i = np.float32(np.random.normal(size=(params['batch_size'],params['dim_z'])))
            train_F(y_0_i, z_i)

            # Train real (meta) discriminator
            y_1_i = np.float32(np.random.normal(loc=1.0,size=(params['batch_size'],1)))
            x_i = iterator.next()[0]
            train_R(y_1_i, x_i)

            # Train Variance Discriminator
            x_i = iterator.next()[0]
            z_i = np.float32(np.random.normal(size=(params['batch_size'],params['dim_z'])))
            D_losses_4[i] = train_D_vral(x_i, z_i)
            D_grad_norms_4[i] = D_grad_fake_wgan_norm_value(z_i)

        # Sample from D
        iterator = stream.get_epoch_iterator()
        for i in range(num_examples / params['batch_size']):
            x_i = iterator.next()[0]
            z_i = np.float32(np.random.normal(size=(params['batch_size'],params['dim_z'])))

            D_samples_regular[i*params['batch_size']:(i+1)*params['batch_size'],0:1] = D_regular_out_R(x_i)
            D_samples_regular[i*params['batch_size']:(i+1)*params['batch_size'],1:2] = D_regular_out_F(z_i)

            D_samples_lsgan[i*params['batch_size']:(i+1)*params['batch_size'],0:1] = D_lsgan_out_R(x_i)
            D_samples_lsgan[i*params['batch_size']:(i+1)*params['batch_size'],1:2] = D_lsgan_out_F(z_i)

            D_samples_wgan[i*params['batch_size']:(i+1)*params['batch_size'],0:1] = D_wgan_out_R(x_i)
            D_samples_wgan[i*params['batch_size']:(i+1)*params['batch_size'],1:2] = D_wgan_out_F(z_i)

            D_samples_vral[i*params['batch_size']:(i+1)*params['batch_size'],0:1] = D_vral_out_R(x_i)
            D_samples_vral[i*params['batch_size']:(i+1)*params['batch_size'],1:2] = D_vral_out_F(z_i)

        # Save Results
        with open(os.path.join(out_dir, 'regular', 'D_samples_%d.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, D_samples_regular)
        with open(os.path.join(out_dir, 'least-squares', 'D_samples_%d.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, D_samples_lsgan)
        with open(os.path.join(out_dir, 'wasserstein', 'D_samples_%d.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, D_samples_wgan)
        with open(os.path.join(out_dir, 'variance', 'D_samples_%d.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, D_samples_vral)

        with open(os.path.join(out_dir, 'regular', '%d_D_loss.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, D_losses_1)
        with open(os.path.join(out_dir, 'least-squares', '%d_D_loss.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, D_losses_2)
        with open(os.path.join(out_dir, 'wasserstein', '%d_D_loss.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, D_losses_3)
        with open(os.path.join(out_dir, 'variance', '%d_D_loss.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, D_losses_4)
        
        with open(os.path.join(out_dir, 'regular', '%d_D_grad_norms.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, D_grad_norms_1)
        with open(os.path.join(out_dir, 'least-squares', '%d_D_grad_norms.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, D_grad_norms_2)
        with open(os.path.join(out_dir, 'wasserstein', '%d_D_grad_norms.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, D_grad_norms_3)
        with open(os.path.join(out_dir, 'variance', '%d_D_grad_norms.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, D_grad_norms_4)
