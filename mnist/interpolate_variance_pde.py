#!/usr/bin/env python

"""
VRAL Interpolation using parzen density estimators.
"""

import numpy as np
from lasagne.layers import get_all_params, get_output, get_all_layers, set_all_param_values
from lasagne.updates import adam
from theano import function
import theano.tensor as T
import os

from data import get_data
from model import discriminator, generator

############################# HELPER FUNCTIONS ################################

def l2(x1, x2):
    """ L2 pixel-wise distance between corresponding images in batches x1, x2. """
    return np.sum((x1 - x2)**2, axis=(1,2,3))

def log_sum_exp(x, axis=None, keepdims=False):
    ''' Code taken from Devon Hjelm. '''
    x_max = T.max(x, axis=axis, keepdims=True)
    y = T.log(T.sum(T.exp(x - x_max), axis=axis, keepdims=True)) + x_max
    y = T.sum(y, axis=axis, keepdims=keepdims)
    return y

###############################################################################

if __name__ == '__main__':
    print('starting')
    params = {
        'adam_beta1':0.5,
        'adam_beta2':0.999,
        'adam_epsilon':3e-6,
        'adam_learning_rate':0.0001,
        'batch_size':64,
        'dim_z':100,
        'iters_D':1,
        'epochs':20,
        'KL': 'inclusive',
        'load_model':False
    }

    out_dir = '/u/grewalka/lasagne/gamma-experiment/variance_pde/inclusive/1_1/'# % (params['KL'], params['iters_D'])
    
    X = T.tensor4()
    z = T.fmatrix()

    D, G = discriminator(X, use_batch_norm=True), generator(z, use_batch_norm=True)
    
    y_real = get_output(D)
    X_fake = get_output(G)
    y_fake = get_output(D, X_fake)

    # Samples from N(0,1) and N(1,1) if using inclusive KL
    y_0 = T.fmatrix()
    y_1 = T.fmatrix()

    # Mixture components for parzen density estimator
    sigma_fake = 1.
    sigma_real = 1.

    # Loss functions
    if params['KL'] == 'inclusive':
        f_mixture = (y_0 - T.shape_padleft(y_fake)) ** 2 / (2.*sigma_fake ** 2)
        r_mixture = (y_1 - T.shape_padleft(y_real)) ** 2 / (2.*sigma_real ** 2) 
        D_loss = -log_sum_exp(-f_mixture, axis=1).mean() - log_sum_exp(-r_mixture, axis=1).mean()
    elif params['KL'] == 'exclusive':
        f_mixture = (T.shape_padleft(y_fake) - (T.tile(T.shape_padaxis(y_fake, 1), (1, y_fake.shape[0]))))
        f_mixture = f_mixture ** 2 / (2.*sigma_fake ** 2)
        r_mixture = (T.shape_padleft(y_real) - (T.tile(T.shape_padaxis(y_real, 1), (1, y_real.shape[0]))))
        r_mixture = r_mixture ** 2 / (2.*sigma_real ** 2)
        D_loss = 0.5 * (y_fake ** 2 + (y_real - 1) ** 2).mean() + log_sum_exp(-f_mixture, axis=1).mean() + log_sum_exp(-r_mixture, axis=1).mean()
    G_loss = 0.5 * ((y_fake.mean() - y_real.mean()) ** 2)

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
        train_D = function([X, z, y_0, y_1], outputs=D_loss, updates=updates_D, allow_input_downcast=True)
    elif params['KL'] == 'exclusive':
        train_D = function([X, z], outputs=D_loss, updates=updates_D, allow_input_downcast=True)
    train_G = function([X, z], outputs=G_loss, updates=updates_G, allow_input_downcast=True)
    
    # Gradient Norms
    D_grad = T.grad(y_real.mean(), X)
    D_grad_norm_value = function([X],outputs=(D_grad**2).sum(axis=(1,2,3)).mean())

    G_loss_value = function([X, X_fake], outputs=G_loss)
    G_loss_grad = T.grad(G_loss, X_fake)
    G_loss_grad_norm_value = function([X, X_fake],outputs=(G_loss_grad**2).sum(axis=(1,2,3)))

    # Sampling functions
    generate = function([z], outputs=X_fake)
    D_out = function([X], outputs=y_real)
    
    D_grad_norms = np.zeros(shape=(200, params['batch_size'], 32))
    D_samples = np.zeros(shape=(200, params['batch_size'], 32))
    G_losses = np.zeros(shape=(200,params['batch_size']))
    G_loss_grad_norms = np.zeros(shape=(200,params['batch_size']))

    # Load data
    stream, num_examples = get_data(params['batch_size'])
    
    if params['load_model']:
        exit(0)
    else:
        with open(os.path.join(out_dir, 'out.log'), 'w+') as f:
            f.write('gamma samples after {} epochs'.format(params['epochs']))
            f.write('{}:1'.format(params['iters_D']))

        print('\nVariance PDE interpolation with D iters = {}'.format(params['iters_D']))
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
                        y_1_i = np.float32(np.tile(np.random.normal(loc=1.0,size=(params['batch_size'])),(params['batch_size'],1)))
                        train_D(x_i, z_i, y_0_i, y_1_i)
                    elif params['KL'] == 'exclusive':
                        train_D(x_i, z_i)
                    
                # train generator
                z_i = np.float32(np.random.normal(size=(params['batch_size'],params['dim_z'])))
                train_G(x_i, z_i)
    
            D_grad_norms.fill(0.)
            D_samples.fill(0.)
    
            # Interpolation
            # Generate pairs
            iterator = stream.get_epoch_iterator()
            x_real = iterator.next()[0]
            
            samples, distances = list(), list()
            for n in range(32):
                z_i = np.float32(np.random.normal(size=(params['batch_size'],params['dim_z'])))
                samples.append(generate(z_i))
                distances.append(l2(x_real, samples[n]))
            
            samples = np.array(samples)
            distances = np.array(distances)
            
            x_fake = list()
            r = np.argmin(distances, axis=0)
            for i in range(64):
                x_fake.append(samples[r[i],i,:,:,:])
            x_fake = np.array(x_fake)

            k = 0
            for gamma in (np.array(range(-50, 150)) / 100.):
        
                # Discriminator output and gradient norms
                for n in range(32):
                    x_hat = np.float32(gamma * x_fake + (1.-gamma) * x_real)
                    D_samples[k,:,n] = D_out(x_hat).reshape(params['batch_size'])
                    D_grad_norms[k,:,n] = D_grad_norm_value(x_hat)
                    
                # Generator loss and objective gradient norm
                x_hat = np.float32(gamma * x_fake + (1.-gamma) * x_real)
                G_losses[k,:] = G_loss_value(x_real, x_hat)
                G_loss_grad_norms[k,:] = G_loss_grad_norm_value(x_real, x_hat)
                k += 1
                
            with open(os.path.join(out_dir, 'D_samples_gamma_{}.npz'.format(epoch+1)), 'w+') as f:
                np.savez(f, D_samples)
            with open(os.path.join(out_dir, 'D_grad_norms_gamma_{}.npz'.format(epoch+1)), 'w+') as f:
                np.savez(f, D_grad_norms)

            with open(os.path.join(out_dir, 'G_loss_gamma_%d.npz' % (epoch+1)), 'w+') as f:
                np.savez(f, G_losses)
            with open(os.path.join(out_dir, 'G_loss_grad_norms_gamma_%d.npz' % (epoch+1)), 'w+') as f:
                np.savez(f, G_loss_grad_norms)
        
            # Save model
            D_params = get_all_params(get_all_layers(D))
            with open(os.path.join(out_dir, 'discriminator_model_{}.npz'.format(params['epochs'])), 'w+') as f:
                np.savez(f, D_params)
            
            G_params = get_all_params(get_all_layers(G))
            with open(os.path.join(out_dir, 'generator_model{}.npz'.format(params['epochs'])), 'w+') as f:
                np.savez(f, G_params)

