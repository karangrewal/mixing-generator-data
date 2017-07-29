#!/usr/bin/env python

"""
VRAL using a single meta-discriminator.
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
        'D_loss': 'proxy',
        'dim_z':100,
        'iters_D':1,
        'iters_M':1,
        'epochs':40,
        'rescale_both':True,
        'wasserstein':2
    }
    
    out_dir = '/u/grewalka/lasagne/variance_single_gaussian/%d_%d_1/' % (params['iters_M'], params['iters_D'])
    
    with open(os.path.join(out_dir, 'out.log'), 'w+') as f:
        f.write('VRAL; single meta-discriminator\n')
        f.write('M iters: {}, D iters: {}'.format(params['iters_M'], params['iters_D']))

    X = T.tensor4()
    z = T.fmatrix()

    D, G = discriminator(X, use_batch_norm=True), generator(z, use_batch_norm=True)

    y_real = get_output(D)
    X_fake = get_output(G)
    y_fake = get_output(D, X_fake)

    if params['rescale_both']:
        y_real_rescaled = y_real - y_real.mean()
    else:
        y_real_rescaled = y_real
    y_fake_rescaled = y_fake - y_fake.mean()

    # Meta-discriminator
    M = real_fake_discriminator(y_fake_rescaled)

    # Samples from N(0,1)
    y_0 = T.fmatrix()

    # Outputs of meta-discriminator
    m_real = get_output(M, y_real_rescaled)
    m_fake = get_output(M)
    m_normal = get_output(M, y_0)

    # Maximize Wasserstein-1 or Wasserstein-2 between y_real and y_fake distributions
    if params['wasserstein'] == 1:
        w_term = y_real.mean() - y_fake.mean()
    elif params['wasserstein'] == 2:
        w_term = T.sqrt((y_real.mean() - y_fake.mean()) ** 2)

    # Loss functions
    M_loss = (T.nnet.softplus(m_normal) + m_real + m_fake + T.nnet.softplus(-m_real) + T.nnet.softplus(-m_fake)).mean()
    if params['D_loss'] == 'regular':
        D_loss = -(m_real + m_fake + T.nnet.softplus(-m_real) + T.nnet.softplus(-m_fake)).mean() - w_term
    elif params['D_loss'] == 'proxy':
        D_loss = (T.nnet.softplus(-m_real) + T.nnet.softplus(-m_fake)).mean() - w_term
    G_loss = -(y_fake).mean()

    # Updates to be performed during training
    updates_M = adam(
        loss_or_grads=M_loss,
        params=get_all_params(M, trainable=True),
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

    train_M = function([y_0, X, z], outputs=M_loss, updates=updates_M, allow_input_downcast=True)
    train_D = function([X, z], outputs=D_loss, updates=updates_D, allow_input_downcast=True)
    train_G = function([z], outputs=G_loss, updates=updates_G, allow_input_downcast=True)

    # Gradient Norms
    M_grad_real = T.grad(m_real.mean(), X)
    M_grad_fake = T.grad(m_fake.mean(), X_fake)
    D_grad_real = T.grad(y_real.mean(), X)
    D_grad_fake = T.grad(y_fake.mean(), X_fake)
    
    # Value of E||dF/dx||^2, E||dR/dx||^2, E||dD/dx||^2
    M_grad_real_norm_value = function([X],outputs=(M_grad_real**2).sum(axis=(1,2,3)).mean())
    M_grad_fake_norm_value = function([z],outputs=(M_grad_fake**2).sum(axis=(1,2,3)).mean())
    D_grad_real_norm_value = function([X],outputs=(D_grad_real**2).sum(axis=(1,2,3)).mean())
    D_grad_fake_norm_value = function([z],outputs=(D_grad_fake**2).sum(axis=(1,2,3)).mean())
    
    # Sampling functions
    generate = function([z], outputs=X_fake)
    D_out_R = function([X], outputs=y_real)
    D_out_F = function([z], outputs=y_fake)

    # Load data
    stream, num_examples = get_data(params['batch_size'])

    print('M iters: {}, D iters: {}'.format(params['iters_M'], params['iters_D']))
    print('Output files will be placed in: {}'.format(out_dir))

    for epoch in range(params['epochs']):
        print('Starting Epoch {}/{} ...'.format(epoch+1, params['epochs']))
        
        # Keep track of Information
        M_losses = np.zeros(shape=(num_examples / params['batch_size'], params['iters_M']))
        D_losses = np.zeros(shape=(num_examples / params['batch_size'], params['iters_D']))
        G_losses = np.zeros(shape=(num_examples / params['batch_size']))

        M_grad_real_norms = np.zeros(shape=(num_examples / params['batch_size'], params['iters_M']))
        M_grad_fake_norms = np.zeros(shape=(num_examples / params['batch_size'], params['iters_M']))
        D_grad_real_norms = np.zeros(shape=(num_examples / params['batch_size'], params['iters_D']))
        D_grad_fake_norms = np.zeros(shape=(num_examples / params['batch_size'], params['iters_D']))
        
        D_samples = np.zeros(shape=((num_examples / params['batch_size'])*params['batch_size'] , 2))
        D_samples.fill(99.)
        
        # Training
        for i in range(num_examples / params['batch_size']):
            
            # Train meta-discriminator
            iterator = stream.get_epoch_iterator()
            for k in range(params['iters_M']):
                
                y_0_i = np.float32(np.random.normal(size=(params['batch_size'],1)))
                x_i = iterator.next()[0]
                z_i = np.float32(np.random.normal(size=(params['batch_size'],params['dim_z'])))
                M_losses[i,0] = train_M(y_0_i, x_i, z_i)
                M_grad_real_norms[i,k] = M_grad_real_norm_value(x_i)
                M_grad_fake_norms[i,k] = M_grad_fake_norm_value(z_i)

            # Train discriminator
            iterator = stream.get_epoch_iterator()
            for k in range(params['iters_D']):

                x_i = iterator.next()[0]
                z_i = np.float32(np.random.normal(size=(params['batch_size'],params['dim_z'])))
                D_losses[i,k] = train_D(x_i, z_i)
                D_grad_real_norms[i,k] = D_grad_real_norm_value(x_i)
                D_grad_fake_norms[i,k] = D_grad_fake_norm_value(z_i)

            # train generator
            z_i = np.float32(np.random.normal(size=(params['batch_size'],params['dim_z'])))
            G_losses[i] = train_G(z_i)

        # Generate samples from G
        z_i = np.float32(np.random.normal(size=(params['batch_size'],params['dim_z'])))
        x_samples = generate(z_i)
        with open(os.path.join(out_dir, 'x_samples_%d.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, x_samples, delimiter=',')

        # Sample from D
        iterator = stream.get_epoch_iterator()
        for i in range(num_examples / params['batch_size']):
            x_i = iterator.next()[0]
            z_i = np.float32(np.random.normal(size=(params['batch_size'],params['dim_z'])))
            D_samples[i*params['batch_size']:(i+1)*params['batch_size'],0:1] = D_out_R(x_i)
            D_samples[i*params['batch_size']:(i+1)*params['batch_size'],1:2] = D_out_F(z_i)

        # Save Results
        with open(os.path.join(out_dir, 'D_samples_%d.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, D_samples, delimiter=',')

        with open(os.path.join(out_dir, '%d_M_loss.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, M_losses, delimiter=',')
        with open(os.path.join(out_dir, '%d_D_loss.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, D_losses, delimiter=',')
        with open(os.path.join(out_dir, '%d_G_loss.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, G_losses, delimiter=',')

        with open(os.path.join(out_dir, '%d_M_grad_real_norms.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, M_grad_real_norms, delimiter=',')
        with open(os.path.join(out_dir, '%d_M_grad_fake_norms.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, M_grad_fake_norms, delimiter=',')
        with open(os.path.join(out_dir, '%d_D_grad_real_norms.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, D_grad_real_norms, delimiter=',')
        with open(os.path.join(out_dir, '%d_D_grad_fake_norms.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, D_grad_fake_norms, delimiter=',')

