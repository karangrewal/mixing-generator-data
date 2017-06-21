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
    # place params in separate file
    params = {
        'adam_beta1':0.5,
        'adam_beta2':0.999,
        'adam_epsilon':3e-6,
        'adam_learning_rate':0.0001,
        'batch_size':64,
        'dim_z':100,
        'iters_D':20,
        'iters_F':20,
        'iters_R':20,
        'epochs':200
    }

    out_dir = '/u/grewalka/cifar10/variance/%d_%d_%d_1/' % (params['iters_F'], params['iters_R'], params['iters_D'])

    with open(os.path.join(out_dir, 'out.log'), 'w+') as f:
        f.write('Variance-maximizing GAN\n')
        f.write('F iters: {}, R iters: {}, D iters: {}'.format(params['iters_F'], params['iters_R'], params['iters_D']))

    X = T.tensor4()
    z = T.fmatrix()

    D, G = discriminator(X), generator(z)

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
    F_grad_fake = T.grad(f_fake.mean(), X_fake)
    R_grad_fake = T.grad(r_fake.mean(), X)
    D_grad_real = T.grad(y_real.mean(), X)
    D_grad_fake = T.grad(y_fake.mean(), X_fake)
    # G_grad = T.grad(G_loss, X_fake)

    # Value of E||dF/dx||^2, E||dR/dx||^2, E||dD/dx||^2
    F_grad_fake_norm_value = function([z],outputs=(F_grad_fake**2).sum(axis=(0,1,2,3)))
    R_grad_fake_norm_value = function([X],outputs=(R_grad_fake**2).sum(axis=(0,1,2,3)))
    D_grad_real_norm_value = function([X],outputs=(D_grad_real**2).sum(axis=(0,1,2,3)))
    D_grad_fake_norm_value = function([z],outputs=(D_grad_fake**2).sum(axis=(0,1,2,3)))
    # G_grad_norm_value = function([z],outputs=(G_grad**2).sum(axis=(0,1,2,3)))

    # Load data
    batches = get_data(params['batch_size'])

    # Sampling functions
    generate = function([z], outputs=X_fake)
    D_out_R = function([X], outputs=y_real)
    D_out_F = function([z], outputs=y_fake)

    print('F iters: {}, R iters: {}, D iters: {}'.format(params['iters_F'], params['iters_R'], params['iters_D']))
    print('Output files will be placed in: {}'.format(out_dir))

    for epoch in range(params['epochs']):
        print('\nStarting Epoch {}/{} ...\n'.format(epoch+1, params['epochs']))
        np.random.shuffle(batches)

        # Keep track of Information
        F_losses = np.zeros(shape=(batches.shape[0], params['iters_F']))
        R_losses = np.zeros(shape=(batches.shape[0], params['iters_R']))
        D_losses = np.zeros(shape=(batches.shape[0], params['iters_D']))
        G_losses = np.zeros(shape=(batches.shape[0]))

        F_grad_fake_norms = np.zeros(shape=(batches.shape[0], params['iters_F']))
        R_grad_fake_norms = np.zeros(shape=(batches.shape[0], params['iters_R']))
        D_grad_real_norms = np.zeros(shape=(batches.shape[0], params['iters_D']))
        D_grad_fake_norms = np.zeros(shape=(batches.shape[0], params['iters_D']))
        # G_grad_norms = np.zeros(shape=(params['epochs'], batches.shape[0]))

        D_samples = np.zeros(shape=(batches.shape[0]*params['batch_size'] , 2))
        D_samples.fill(99.)

        # Training
        for i in range(batches.shape[0]):

            # Train fake data discriminator
            for k in range(params['iters_F']):
                v_0_i = np.float32(np.random.normal(size=(params['batch_size'],1)))
                z_i = np.float32(np.random.normal(size=(params['batch_size'],params['dim_z'])))
                F_losses[i,k] = train_F(v_0_i, z_i)
                F_grad_fake_norms[i,k] = F_grad_fake_norm_value(z_i)

            # Train real data discriminator
            for k in range(params['iters_R']):
                v_1_i = np.float32(np.random.normal(loc=1.0,size=(params['batch_size'],1)))
                x_i = batches[np.random.randint(batches.shape[0])]
                R_losses[i,k] = train_R(v_1_i, x_i)
                R_grad_fake_norms[i,k] = R_grad_fake_norm_value(x_i)

            # Train discriminator
            for k in range(params['iters_D']):
                np.random.shuffle(batches)
                x_i = batches[np.random.randint(batches.shape[0])]
                z_i = np.float32(np.random.normal(size=(params['batch_size'],params['dim_z'])))
                D_losses[i,k] = train_D(x_i, z_i)
                D_grad_real_norms[i,k] = D_grad_real_norm_value(x_i)
                D_grad_fake_norms[i,k] = D_grad_fake_norm_value(z_i)

            # train generator
            z_i = np.float32(np.random.normal(size=(params['batch_size'],params['dim_z'])))
            G_losses[i] = train_G(z_i)
            # G_grad_norms[i] = G_grad_norm_value(z_i)

        if (epoch + 1) % 10 == 0:
            # Generate samples from G
            z_i = np.float32(np.random.normal(size=(params['batch_size'],params['dim_z'])))
            x_samples = generate(z_i)
            with open(os.path.join(out_dir, 'x_samples_%d.npz' % (epoch+1)), 'w+') as f:
                np.savez(f, x_samples, delimiter=',')

            # Sample from D
            for i in range(batches.shape[0]):
                x_i = batches[i]
                z_i = np.float32(np.random.normal(size=(params['batch_size'],params['dim_z'])))
                D_samples[i*params['batch_size']:(i+1)*params['batch_size'],0:1] = D_out_R(x_i)
                D_samples[i*params['batch_size']:(i+1)*params['batch_size'],1:2] = D_out_F(z_i)

            # Save Results
            with open(os.path.join(out_dir, 'D_samples_%d.npz' % (epoch+1)), 'w+') as f:
                np.savez(f, D_samples, delimiter=',')

            with open(os.path.join(out_dir, '%d_F_loss.npz' % (epoch+1)), 'w+') as f:
                np.savez(f, F_losses, delimiter=',')
            with open(os.path.join(out_dir, '%d_R_loss.npz' % (epoch+1)), 'w+') as f:
                np.savez(f, R_losses, delimiter=',')
            with open(os.path.join(out_dir, '%d_D_loss.npz' % (epoch+1)), 'w+') as f:
                np.savez(f, D_losses, delimiter=',')
            with open(os.path.join(out_dir, '%d_G_loss.npz' % (epoch+1)), 'w+') as f:
                np.savez(f, G_losses, delimiter=',')

            with open(os.path.join(out_dir, '%d_F_grad_fake_norms.npz' % (epoch+1)), 'w+') as f:
                np.savez(f, F_grad_fake_norms, delimiter=',')
            with open(os.path.join(out_dir, '%d_R_grad_fake_norms.npz' % (epoch+1)), 'w+') as f:
                np.savez(f, R_grad_fake_norms, delimiter=',')
            with open(os.path.join(out_dir, '%d_D_grad_real_norms.npz' % (epoch+1)), 'w+') as f:
                np.savez(f, D_grad_real_norms, delimiter=',')
            with open(os.path.join(out_dir, '%d_D_grad_fake_norms.npz' % (epoch+1)), 'w+') as f:
                np.savez(f, D_grad_fake_norms, delimiter=',')
            # with open(os.path.join(out_dir, 'G_grad_norms.npz'), 'w+') as f:
            #     np.savez(f, G_grad_norms, delimiter=',')

    # Save Generator
    G_params = get_all_params(get_all_layers(G))
    with open(os,path.join(out_dir, 'generator_model.npz')) as f:
        np.savez(f, G_params, delimiter=',')
