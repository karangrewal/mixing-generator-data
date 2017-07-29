#!/usr/bin/env python

"""
VRAL Interpolation: (1) rescaled and fitted to a single gaussian (2) parzen density estimators

KL:
    inclusive : KL( N(.,1) || p(x) )
    exclusive : KL( p(x) || N(.,1) )

G_loss_type:
    1: G's objective is to maximize E_{x~P_fake} [D(x)]
    2: G's objective is to minimize ( E_{x~P_fake} [D(x)] - E_{x~P_real} [D(x)])^2
"""

import numpy as np
from lasagne.layers import get_all_params, get_output, get_all_layers, set_all_param_values
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
    # place params in separate file
    params = {
        'adam_beta1':0.5,
        'adam_beta2':0.999,
        'adam_epsilon':3e-6,
        'adam_learning_rate':0.0001,
        'batch_size':64,
        'dim_z':100,
        'G_loss_type':1,
        'iters_D':1,
        'epochs':20,
        'KL': 'inclusive',
        'load_model':False,
        'rescale_both':False,
        'wasserstein':1
    }

    out_dir = '/u/grewalka/lasagne/gamma-experiment/variance_pde_sg/%s/%d_1/' % (params['KL'], params['iters_D'])
    
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
    D_grad = T.grad(y_real.mean(), X)
    D_grad_norm_value = function([X],outputs=(D_grad**2).sum(axis=(1,2,3)).mean())

    # Load data
    stream, num_examples = get_data(params['batch_size'])

    # Sampling functions
    generate = function([z], outputs=X_fake)
    D_out = function([X], outputs=y_real)
    
    D_grad_norms = np.zeros(shape=(200, 32))
    D_samples = np.zeros(shape=(200, params['batch_size'], 32))
    
    if params['load_model']:
        
        # Load parameters
        with open(os.path.join(out_dir, 'discriminator_model_{}.npz'.format(params['epochs']))) as f:
            D_params = np.load(f)['arr_0']
        set_all_param_values(D, [param.get_value() for param in D_params])
        
        with open(os.path.join(out_dir, 'generator_model_{}.npz'.format(params['epochs']))) as f:
            G_params = np.load(f)['arr_0']
        set_all_param_values(G, [param.get_value() for param in G_params])
        
        k = 0
        for gamma in (np.array(range(-50, 150)) / 100.):
    
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
            f.write('{}:1'.format(params['iters_D']))

        print('D iters: {}'.format(params['iters_D']))
        print('Output files will be placed in: {}'.format(out_dir))
    
        for epoch in range(params['epochs']):
            print('Starting Epoch {}/{} ...'.format(epoch+1, params['epochs']))
    
            # Training
            for i in range(num_examples / params['batch_size']):
    
                 # Train fake data discriminator
                iterator = stream.get_epoch_iterator()
                for k in range(params['iters_D']):
    
                    # Train discriminator
                    x_i = iterator.next()[0]
                    z_i = np.float32(np.random.normal(size=(params['batch_size'],params['dim_z'])))
                    if params['KL'] == 'inclusive':
                        y_0_i = np.float32(np.tile(np.random.normal(size=(params['batch_size'])),(params['batch_size'],1)))
                        train_D(x_i, z_i, y_0_i)
                    elif params['KL'] == 'exclusive':
                        train_D(x_i, z_i)
                    
                # train generator
                z_i = np.float32(np.random.normal(size=(params['batch_size'],params['dim_z'])))
                if params['G_loss_type'] == 1:
                    train_G(z_i)
                elif params['G_loss_type'] == 2:
                    train_G(x_i, z_i)
    
            D_grad_norms.fill(0.)
            D_samples.fill(0.)
    
            # **
            # Interpolate
            # **
            k = 0
            for gamma in (np.array(range(-50, 150)) / 100.):
        
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
        with open(os.path.join(out_dir, 'discriminator_model_{}.npz'.format(params['epochs'])), 'w+') as f:
            np.savez(f, D_params)
        
        G_params = get_all_params(get_all_layers(G))
        with open(os.path.join(out_dir, 'generator_model_{}.npz'.format(params['epochs'])), 'w+') as f:
            np.savez(f, G_params)



