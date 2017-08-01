#!/usr/bin/env python

import numpy as np
from lasagne.layers import get_all_params, get_output, get_all_layers, set_all_param_values
from theano import function
import theano.tensor as T
import os

from model import low_cap_discriminator, high_cap_discriminator, discriminator, generator
from data import get_data

def l2(x1, x2):
    """
    L2 pixel-wise distance between corresponding images in batches x1, x2.
    """
    return np.sum((x1 - x2)**2, axis=(1,2,3))

if __name__ == '__main__':
    params = {
        'batch_size':64,
        'dim_z':100,
        'epochs':20,
    }

    out_dir = ''
    
    X = T.tensor4()
    z = T.fmatrix()

    D, G = discriminator(X, use_batch_norm=True), generator(z, use_batch_norm=True)

    y_real = get_output(D)
    X_fake = get_output(G)
    y_fake = get_output(D, X_fake)
    
    # Regular GAN loss
    if 'regular' in out_dir:
        G_loss = (-(y_fake + T.nnet.softplus(-y_fake))).mean()
    elif 'variance' in out_dir or 'least-squares' in out_dir:
        G_loss = 0.5 * ((y_fake - 1) ** 2).mean()
    elif 'wasserstein' in out_dir:
        G_loss = (-y_fake).mean()
    elif 'proxy' in out_dir:
        G_loss = (T.nnet.softplus(-y_fake)).mean()
    else:
        exit(0)

    # Gradient Norms
    D_grad = T.grad(y_real.mean(), X)
    D_grad_norm_value = function([X],outputs=(D_grad**2).sum(axis=(1,2,3)))
    
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
    
    # Load parameters
    with open(os.path.join(out_dir, 'discriminator_model_{}.npz'.format(params['epochs']))) as f:
        D_params = np.load(f)['arr_0']
    set_all_param_values(D, [param.get_value() for param in D_params])
    
    with open(os.path.join(out_dir, 'generator_model_{}.npz'.format(params['epochs']))) as f:
        G_params = np.load(f)['arr_0']
    set_all_param_values(G, [param.get_value() for param in G_params])

    D_grad_norms.fill(0.)
    D_samples.fill(0.)

    # Generate pairs
    k = 0
    stream, num_examples = get_data(params['batch_size'])
    iterator = stream.get_epoch_iterator()
    x_real = iterator.next()[0]

    samples, distances = list(), list()
    for n in range(32):
        z_i = np.float32(np.random.normal(size=(params['batch_size'],params['dim_z'])))
        samples.append(generate(z_i))
        distances.append(l2(x_real, samples[n]))

    samples = np.array(samples)
    distances = np.array(distances)

    best = list()
    r = np.argmin(distances, axis=0)
    for i in range(64):
        best.append(samples[r[i],i,:,:,:])

    x_fake = np.array(best)

    # Interpolate
    # k indexes the value of gamma
    # n indexes the batch (only 1 batch is necessary)
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
        
    with open(os.path.join(out_dir, 'D_samples_gamma_star2.npz'), 'w+') as f:
        np.savez(f, D_samples)
    with open(os.path.join(out_dir, 'D_grad_norms_gamma_star2.npz'), 'w+') as f:
        np.savez(f, D_grad_norms)
    
    with open(os.path.join(out_dir, 'G_loss_gamma_star2.npz'), 'w+') as f:
        np.savez(f, G_losses)
    with open(os.path.join(out_dir, 'G_loss_grad_norms_gamma_star2.npz'), 'w+') as f:
        np.savez(f, G_loss_grad_norms)
    