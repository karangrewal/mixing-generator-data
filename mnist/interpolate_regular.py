#!/usr/bin/env python

"""
Interpolate Regular GAN.
"""

import numpy as np
from lasagne.layers import get_all_params, get_output, get_all_layers, set_all_param_values
from lasagne.updates import adam
from theano import function
import theano.tensor as T
import os

from data import get_data
from model import discriminator, generator

def l2(x1, x2):
    """ L2 pixel-wise distance between corresponding images in batches x1, x2. """
    return np.sum((x1 - x2)**2, axis=(1,2,3))

if __name__ == '__main__':
    print('starting')
    params = {
        'adam_beta1':0.5,
        'adam_beta2':0.999,
        'adam_epsilon':3e-6,
        'adam_learning_rate':0.0001,
        'batch_size':64,
        'dim_z':100,
        'discriminator_iters':1,
        'epochs':20,
        'load_model':False
    }

    out_dir = '/u/grewalka/lasagne/gamma-experiment/regular/bn_g/1_1/'# % (params['discriminator_iters'])
    
    X = T.tensor4()
    z = T.fmatrix()

    D, G = discriminator(X, use_batch_norm=False), generator(z, use_batch_norm=True)

    y_real = get_output(D)
    X_fake = get_output(G)
    y_fake = get_output(D, X_fake)
    
    D_loss = (y_fake + T.nnet.softplus(-y_real) + T.nnet.softplus(-y_fake)).mean()
    G_loss = (-(y_fake + T.nnet.softplus(-y_fake))).mean()

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

    train_D = function([X, z], outputs=D_loss, updates=updates_D, allow_input_downcast=True)
    train_G = function([z], outputs=G_loss, updates=updates_G, allow_input_downcast=True)
    
    # Gradient Norms
    D_grad = T.grad(y_real.mean(), X)
    D_grad_norm_value = function([X],outputs=(D_grad**2).sum(axis=(1,2,3)).mean())
    
    G_loss_value = function([X_fake], outputs=G_loss)
    G_loss_grad = T.grad(G_loss, X_fake)
    G_loss_grad_norm_value = function([X_fake],outputs=(G_loss_grad**2).sum(axis=(1,2,3)))

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
            f.write('{}:1'.format(params['discriminator_iters']))

        print('D_iters: {}'.format(params['discriminator_iters']))
        print('Output files will be placed in: {}'.format(out_dir))
    
        for epoch in range(params['epochs']):
            print('Starting Epoch {}/{} ...'.format(epoch+1, params['epochs']))
    
            # Training
            for i in range(num_examples / params['batch_size']):
    
                 # Train fake data discriminator
                iterator = stream.get_epoch_iterator()
                for k in range(params['discriminator_iters']):
    
                    # Train discriminator
                    x_i = iterator.next()[0]
                    z_i = np.float32(np.random.normal(size=(params['batch_size'],params['dim_z'])))
                    train_D(x_i, z_i)
                    
                # train generator
                z_i = np.float32(np.random.normal(size=(params['batch_size'],params['dim_z'])))
                train_G(z_i)
    
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
                G_losses[k,:] = G_loss_value(x_hat)
                G_loss_grad_norms[k,:] = G_loss_grad_norm_value(x_hat)
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
            with open(os.path.join(out_dir, 'discriminator_model_{}.npz'.format(epoch+1)), 'w+') as f:
                np.savez(f, D_params)
            
            G_params = get_all_params(get_all_layers(G))
            with open(os.path.join(out_dir, 'generator_model_{}.npz'.format(epoch+1)), 'w+') as f:
                np.savez(f, G_params)
