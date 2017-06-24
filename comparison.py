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
    # place params in separate file
    params = {
        'adam_beta1':0.5,
        'adam_beta2':0.999,
        'adam_epsilon':3e-6,
        'adam_learning_rate':0.0001,
        'batch_size':64,
        'dim_z':100,
        'iters_G':1,
        'epochs':200,
        'load_generator':False
    }

    out_dir = '/u/grewalka/lasagne/comparison/gen_%d/' % (params['iters_G'])

    ###########################################################################
    ########################### [1] Train Generator ###########################
    ###########################################################################

    X, z = T.tensor4(), T.fmatrix()
    G = generator(z)
    
    if params['load_generator']:
        with open(os.path.join(out_dir, 'generator_model.npz')) as f:
            G_params = np.load(f)['arr_0']
        set_all_param_values(G, [param.get_value() for param in G_params])
    else:
        D = discriminator(X)
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
    
        # Load data
        batches = get_data(params['batch_size'])
    
        # Sampling functions
        generate = function([z], outputs=X_fake)
    
        print('Output files will be placed in: {}'.format(out_dir))
    
        for epoch in range(params['iters_G']):
            print('\nTraining Generator. Epoch {}/20 ...\n'.format(epoch+1))
            np.random.shuffle(batches)
    
            # Training
            for i in range(batches.shape[0]):
    
                # traing discriminator
                np.random.shuffle(batches)
                x_i = batches[np.random.randint(batches.shape[0])]
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
            exit(0)
        # Save Generator
        G_params = get_all_params(get_all_layers(G))
        with open(os.path.join(out_dir, 'generator_model.npz'), 'w+') as f:
            np.savez(f, G_params, delimiter=',')

    ###########################################################################
    ############## [2] Train Discriminators, Hold Generator Fixed #############
    ###########################################################################

    X_regular, z_regular = T.tensor4(), T.fmatrix()
    X_lsgan, z_lsgan = T.tensor4(), T.fmatrix()
    X_wgan, z_wgan = T.tensor4(), T.fmatrix()
    X_variance, z_variance = T.tensor4(), T.fmatrix()
    
    # Discriminators
    D_regular, D_lsgan, D_wgan, D_variance = discriminator(X_regular), discriminator(X_lsgan), discriminator(X_wgan), discriminator(X_variance)
    
    y_real_regular, X_fake_regular = get_output(D_regular), get_output(G)
    y_real_lsgan, X_fake_lsgan = get_output(D_lsgan), get_output(G)
    y_real_wgan, X_fake_wgan = get_output(D_wgan), get_output(G)
    y_real_variance, X_fake_variance = get_output(D_variance), get_output(G)
    
    y_fake_regular = get_output(D_regular, X_fake_regular)
    y_fake_lsgan = get_output(D_lsgan, X_fake_lsgan)
    y_fake_wgan = get_output(D_wgan, X_fake_wgan)
    y_fake_variance = get_output(D_variance, X_fake_variance)
    
    F, R = real_fake_discriminator(y_fake_variance), real_fake_discriminator(y_real_variance)

    # Samples from N(0,1) and N(1,1)
    v_0, v_1 = T.fmatrix(), T.fmatrix()

    # Outputs of real and fake discriminators
    r_real, r_fake = get_output(R, v_1), get_output(R)
    f_real, f_fake = get_output(F, v_0), get_output(F)

    # Loss Functions
    D_loss_regular = (y_fake_regular + T.nnet.softplus(-y_real_regular) + T.nnet.softplus(-y_fake_regular)).mean()
    D_loss_lsgan = 0.5 * ((y_real_lsgan - 1) ** 2).mean() + 0.5 * ((y_fake_lsgan + 1) ** 2).mean()
    D_loss_wgan = (y_fake_wgan - y_real_wgan).mean()
    
    D_loss_variance = (T.nnet.softplus(-f_fake) + T.nnet.softplus(-r_fake)).mean()
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
    
    updates_D_variance = adam(loss_or_grads=D_loss_variance,
        params=get_all_params(D_variance, trainable=True),
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

    train_D_variance = function([X_variance, z], outputs=D_loss_variance,
        updates=updates_D_variance, allow_input_downcast=True)
    train_F = function([v_0, z], outputs=F_loss, updates=updates_F,
        allow_input_downcast=True)
    train_R = function([v_1, X_variance], outputs=R_loss, updates=updates_R,
        allow_input_downcast=True)

    # Gradient Norms
    D_grad_fake_regular = T.grad(y_fake_regular.mean(), X_fake_regular)
    D_grad_fake_lsgan = T.grad(y_fake_lsgan.mean(), X_fake_lsgan)
    D_grad_fake_wgan = T.grad(y_fake_wgan.mean(), X_fake_wgan)

    D_grad_fake_variance = T.grad(y_fake_variance.mean(), X_fake_variance)
    F_grad_fake = T.grad(f_fake.mean(), X_fake_variance)
    R_grad_fake = T.grad(r_fake.mean(), X_variance)

    # Gradient Norm Functions
    D_grad_fake_regular_norm_value = function([z],outputs=(D_grad_fake_regular**2).sum(axis=(0,1,2,3)))
    D_grad_fake_lsgan_norm_value = function([z],outputs=(D_grad_fake_lsgan**2).sum(axis=(0,1,2,3)))
    D_grad_fake_wgan_norm_value = function([z],outputs=(D_grad_fake_wgan**2).sum(axis=(0,1,2,3)))
    
    D_grad_fake_variance_norm_value = function([z],outputs=(D_grad_fake_variance**2).sum(axis=(0,1,2,3)))
    F_grad_fake_norm_value = function([z],outputs=(F_grad_fake**2).sum(axis=(0,1,2,3)))
    R_grad_fake_norm_value = function([X_variance],outputs=(R_grad_fake**2).sum(axis=(0,1,2,3)))

    # Sampling functions
    D_regular_out_R = function([X_regular], outputs=y_real_regular)
    D_regular_out_F = function([z], outputs=y_fake_regular)

    D_lsgan_out_R = function([X_lsgan], outputs=y_real_lsgan)
    D_lsgan_out_F = function([z], outputs=y_fake_lsgan)

    D_wgan_out_R = function([X_wgan], outputs=y_real_wgan)
    D_wgan_out_F = function([z], outputs=y_fake_wgan)

    D_variance_out_R = function([X_variance], outputs=y_real_variance)
    D_variance_out_F = function([z], outputs=y_fake_variance)

    # Load data
    batches = get_data(params['batch_size'])

    for epoch in range(params['epochs']):
        print('\nStarting Epoch {}/{} ...\n'.format(epoch+1, params['epochs']))
        np.random.shuffle(batches)

        # Keep track of Information
        # Losses
        D_losses_1 = np.zeros(shape=(batches.shape[0]))
        D_losses_2 = np.zeros(shape=(batches.shape[0]))
        D_losses_3 = np.zeros(shape=(batches.shape[0]))

        D_losses_4 = np.zeros(shape=(batches.shape[0]))
        F_losses = np.zeros(shape=(batches.shape[0]))
        R_losses = np.zeros(shape=(batches.shape[0]))

        # Gradient Norms
        D_grad_norms_1 = np.zeros(shape=(batches.shape[0]))
        D_grad_norms_2 = np.zeros(shape=(batches.shape[0]))
        D_grad_norms_3 = np.zeros(shape=(batches.shape[0]))

        D_grad_norms_4 = np.zeros(shape=(batches.shape[0]))
        F_grad_fake_norms = np.zeros(shape=(batches.shape[0]))
        R_grad_fake_norms = np.zeros(shape=(batches.shape[0]))

        # Output Distributions
        D_samples_regular = np.zeros(shape=(batches.shape[0]*params['batch_size'], 2))
        D_samples_lsgan = np.zeros(shape=(batches.shape[0]*params['batch_size'], 2))
        D_samples_wgan = np.zeros(shape=(batches.shape[0]*params['batch_size'], 2))
        D_samples_variance = np.zeros(shape=(batches.shape[0]*params['batch_size'], 2))

        # Training
        for i in range(batches.shape[0]):

            # Train Regular Discriminator
            np.random.shuffle(batches)
            x_i = batches[np.random.randint(batches.shape[0])]
            z_i = np.float32(np.random.normal(size=(params['batch_size'],params['dim_z'])))
            D_losses_1[i] = train_D_regular(x_i, z_i)
            D_grad_norms_1[i] = D_grad_fake_regular_norm_value(z_i)

            # Train Least Squares Discriminator
            np.random.shuffle(batches)
            x_i = batches[np.random.randint(batches.shape[0])]
            z_i = np.float32(np.random.normal(size=(params['batch_size'],params['dim_z'])))
            D_losses_2[i] = train_D_lsgan(x_i, z_i)
            D_grad_norms_2[i] = D_grad_fake_lsgan_norm_value(z_i)

            # Train Regular Discriminator
            np.random.shuffle(batches)
            x_i = batches[np.random.randint(batches.shape[0])]
            z_i = np.float32(np.random.normal(size=(params['batch_size'],params['dim_z'])))
            D_losses_3[i] = train_D_wgan(x_i, z_i)
            D_grad_norms_3[i] = D_grad_fake_wgan_norm_value(z_i)

            # Train fake (meta) discriminator
            v_0_i = np.float32(np.random.normal(size=(params['batch_size'],1)))
            z_i = np.float32(np.random.normal(size=(params['batch_size'],params['dim_z'])))
            F_losses[i] = train_F(v_0_i, z_i)
            F_grad_fake_norms[i] = F_grad_fake_norm_value(z_i)

            # Train real (meta) discriminator
            np.random.shuffle(batches)
            v_1_i = np.float32(np.random.normal(loc=1.0,size=(params['batch_size'],1)))
            x_i = batches[np.random.randint(batches.shape[0])]
            R_losses[i] = train_R(v_1_i, x_i)
            R_grad_fake_norms[i] = R_grad_fake_norm_value(x_i)

            # Train Variance Discriminator
            np.random.shuffle(batches)
            x_i = batches[np.random.randint(batches.shape[0])]
            z_i = np.float32(np.random.normal(size=(params['batch_size'],params['dim_z'])))
            D_losses_4[i] = train_D_wgan(x_i, z_i)
            D_grad_norms_4[i] = D_grad_fake_wgan_norm_value(z_i)

        # Sample from D
        for i in range(batches.shape[0]):
            x_i = batches[i]
            z_i = np.float32(np.random.normal(size=(params['batch_size'],params['dim_z'])))

            D_samples_regular[i*params['batch_size']:(i+1)*params['batch_size'],0:1] = D_regular_out_R(x_i)
            D_samples_regular[i*params['batch_size']:(i+1)*params['batch_size'],1:2] = D_regular_out_F(z_i)

            D_samples_lsgan[i*params['batch_size']:(i+1)*params['batch_size'],0:1] = D_lsgan_out_R(x_i)
            D_samples_lsgan[i*params['batch_size']:(i+1)*params['batch_size'],1:2] = D_lsgan_out_F(z_i)

            D_samples_wgan[i*params['batch_size']:(i+1)*params['batch_size'],0:1] = D_wgan_out_R(x_i)
            D_samples_wgan[i*params['batch_size']:(i+1)*params['batch_size'],1:2] = D_wgan_out_F(z_i)

            D_samples_variance[i*params['batch_size']:(i+1)*params['batch_size'],0:1] = D_variance_out_R(x_i)
            D_samples_variance[i*params['batch_size']:(i+1)*params['batch_size'],1:2] = D_variance_out_F(z_i)

        # Save Results
        with open(os.path.join(out_dir, 'regular', 'D_samples_%d.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, D_samples_regular)
        with open(os.path.join(out_dir, 'least-squares', 'D_samples_%d.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, D_samples_lsgan)
        with open(os.path.join(out_dir, 'wasserstein', 'D_samples_%d.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, D_samples_wgan)
        with open(os.path.join(out_dir, 'variance', 'D_samples_%d.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, D_samples_variance)

        with open(os.path.join(out_dir, 'regular', '%d_D_loss.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, D_losses_1)
        with open(os.path.join(out_dir, 'least-squares', '%d_D_loss.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, D_losses_2)
        with open(os.path.join(out_dir, 'wasserstein', '%d_D_loss.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, D_losses_3)
        with open(os.path.join(out_dir, 'variance', '%d_D_loss.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, D_losses_4)
        with open(os.path.join(out_dir, 'variance', '%d_F_loss.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, F_losses)
        with open(os.path.join(out_dir, 'variance', '%d_R_loss.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, R_losses)

        with open(os.path.join(out_dir, 'regular', '%d_D_grad_norms.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, D_grad_norms_1)
        with open(os.path.join(out_dir, 'least-squares', '%d_D_grad_norms.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, D_grad_norms_2)
        with open(os.path.join(out_dir, 'wasserstein', '%d_D_grad_norms.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, D_grad_norms_3)
        with open(os.path.join(out_dir, 'variance', '%d_D_grad_norms.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, D_grad_norms_4)
        with open(os.path.join(out_dir, 'variance', '%d_F_grad_norms.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, F_grad_fake_norms)
        with open(os.path.join(out_dir, 'variance', '%d_R_grad_norms.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, R_grad_fake_norms)
