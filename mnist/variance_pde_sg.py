#!/usr/bin/env python

"""
VRAL using parzen density estimation for a single gaussian.
    
KL: either 'inclusive' or 'exclusive'
    inclusive : KL( N(.,1) || p(x) )
    exclusive : KL( p(x) || N(.,1) )

G_loss_type:
    1: G's objective is to maximize E_{x~P_fake} [D(x)]
    2: G's objective is to minimize ( E_{x~P_fake} [D(x)] - E_{x~P_real} [D(x)])^2
"""

import numpy as np
from lasagne.layers import get_all_params, get_output, get_all_layers
from lasagne.updates import adam
from theano import function
import theano.tensor as T
import os

from data import get_data
from model import discriminator, generator

# HELPER FUNCTION
def log_sum_exp(x, axis=None, keepdims=False):
    ''' Code taken from Devon Hjelm. '''
    x_max = T.max(x, axis=axis, keepdims=True)
    y = T.log(T.sum(T.exp(x - x_max), axis=axis, keepdims=True)) + x_max
    y = T.sum(y, axis=axis, keepdims=keepdims)
    return y

if __name__ == '__main__':
    params = {
        'adam_beta1':0.5,
        'adam_beta2':0.999,
        'adam_epsilon':3e-6,
        'adam_learning_rate':0.0001,
        'batch_size':64,
        'dim_z':100,
        'G_loss_type':1,
        'iters_D':1,
        'epochs':60,
        'KL': 'inclusive',
        'rescale_both':False,
        'wasserstein':1
    }

    out_dir = '/u/grewalka/lasagne/variance_pde_sg/%s/%d_1/' % (params['KL'], params['iters_D'])
    
    with open(os.path.join(out_dir, 'out.log'), 'w+') as f:
        f.write('VRAL with Parzen Density Estimator, fitting to Single Gaussian\n')
        f.write('D iters: {}'.format(params['iters_D']))

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

    # Samples from N(0,1) if using inclusive KL
    y_0 = T.fmatrix()

    # Mixture components for parzen density estimator
    sigma_fake = 1.
    sigma_real = 1.

    # Maximize Wasserstein-1 or Wasserstein-2 between y_real and y_fake distributions
    if params['wasserstein'] == 1:
        w_term = y_real.mean() - y_fake.mean()
    elif params['wasserstein'] == 2:
        w_term = T.sqrt((y_real.mean() - y_fake.mean()) ** 2)

    # Loss functions
    if params['KL'] == 'inclusive':
        f_mixture = (y_0 - T.shape_padleft(y_fake_rescaled)) ** 2 / (2.*sigma_fake ** 2)
        r_mixture = (y_0 - T.shape_padleft(y_real_rescaled)) ** 2 / (2.*sigma_real ** 2) 
        D_loss = -log_sum_exp(-f_mixture, axis=1).mean() - log_sum_exp(-r_mixture, axis=1).mean() - w_term
    elif params['KL'] == 'exclusive':
        print('P.D.E. with Single Gaussian, Exclusive KL loss function not available.')
        exit(0)

    if params['G_loss_type'] == 1:
        G_loss = -(y_fake).mean()
    elif params['G_loss_type'] == 2:
        G_loss = (y_fake.mean() - y_real.mean()) ** 2

    # Updates to be performed during training
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

    if params['KL'] == 'inclusive':
        train_D = function([X, z, y_0], outputs=D_loss, updates=updates_D, allow_input_downcast=True)
    elif params['KL'] == 'exclusive':
        train_D = function([X, z], outputs=D_loss, updates=updates_D, allow_input_downcast=True)
    
    if params['G_loss_type'] == 1:
        train_G = function([z], outputs=G_loss, updates=updates_G, allow_input_downcast=True)
    elif params['G_loss_type'] == 2:
        train_G = function([X, z], outputs=G_loss, updates=updates_G, allow_input_downcast=True)

    # Gradient Norms
    D_grad_real = T.grad(y_real.mean(), X)
    D_grad_fake = T.grad(y_fake.mean(), X_fake)

    # Value of E||dF/dx||^2, E||dR/dx||^2, E||dD/dx||^2
    D_grad_real_norm_value = function([X],outputs=(D_grad_real**2).sum(axis=(1,2,3)).mean())
    D_grad_fake_norm_value = function([z],outputs=(D_grad_fake**2).sum(axis=(1,2,3)).mean())
    
    # Load data
    stream, num_examples = get_data(params['batch_size'])

    # Sampling functions
    generate = function([z], outputs=X_fake)
    D_out_R = function([X], outputs=y_real)
    D_out_F = function([z], outputs=y_fake)

    print('D iters: {}'.format(params['iters_D']))
    print('Output files will be placed in: {}'.format(out_dir))

    for epoch in range(params['epochs']):
        print('Starting Epoch {}/{} ...'.format(epoch+1, params['epochs']))
        
        # Keep track of Information
        D_losses = np.zeros(shape=(num_examples / params['batch_size'], params['iters_D']))
        G_losses = np.zeros(shape=(num_examples / params['batch_size']))

        D_grad_real_norms = np.zeros(shape=(num_examples / params['batch_size'], params['iters_D']))
        D_grad_fake_norms = np.zeros(shape=(num_examples / params['batch_size'], params['iters_D']))
        # G_grad_norms = np.zeros(shape=(params['epochs'], batches.shape[0]))

        D_samples = np.zeros(shape=((num_examples / params['batch_size'])*params['batch_size'] , 2))
        D_samples.fill(99.)

        # Training
        for i in range(num_examples / params['batch_size']):

            # Train fake data discriminator
            iterator = stream.get_epoch_iterator()
            for k in range(params['iters_D']):

                # Train discriminator
                x_i = iterator.next()[0]
                z_i = np.float32(np.random.normal(size=(params['batch_size'],params['dim_z'])))
                y_0_i = np.float32(np.tile(np.random.normal(size=(params['batch_size'])),(params['batch_size'],1)))
                
                if params['KL'] == 'inclusive':
                    D_losses[i,k] = train_D(x_i, z_i, y_0_i)
                elif params['KL'] == 'exclusive':
                    D_losses[i,k] = train_D(x_i, z_i)
                D_grad_real_norms[i,k] = D_grad_real_norm_value(x_i)
                D_grad_fake_norms[i,k] = D_grad_fake_norm_value(z_i)

            # train generator
            z_i = np.float32(np.random.normal(size=(params['batch_size'],params['dim_z'])))
            if params['G_loss_type'] == 1:
                G_losses[i] = train_G(z_i)
            elif params['G_loss_type'] == 2:
                G_losses[i] = train_G(x_i, z_i)
            
        # Generate samples from G
        z_i = np.float32(np.random.normal(size=(params['batch_size'],params['dim_z'])))
        x_samples = generate(z_i)
        with open(os.path.join(out_dir, 'x_samples_%d.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, x_samples, delimiter=',')

        # Distribution over D
        iterator = stream.get_epoch_iterator()
        for i in range(num_examples / params['batch_size']):
            x_i = iterator.next()[0]
            z_i = np.float32(np.random.normal(size=(params['batch_size'],params['dim_z'])))
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

