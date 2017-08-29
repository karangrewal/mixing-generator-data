egular and Proxy GAN for CelebA. """

import numpy as np
from lasagne.layers import get_all_params, get_output, get_all_layers
from lasagne.updates import adam
from theano import function
import theano.tensor as T
import os

from data import get_data
from model import discriminator, generator

# BATCH NORM PARAMS
BN_D = False
BN_G = False

# HYPERPARAMS
ADAM_BETA1 = 0.1
ADAM_BETA2 = 0.999
ADAM_EPSILON = 3e-6
ADAM_LR = 0.0001
BATCH_SIZE = 64
DIM_C = 3
DIM_X = 64
DIM_Y = 64
DIM_Z = 128
EPOCHS = 100
ITERS_D = 1
METHOD = 'proxy'

if __name__ == '__main__':
    print('starting')
    if METHOD == 'proxy':
        if BN_D and BN_G:
            out_dir = '/u/grewalka/celeba/proxy/%d_1/' % (ITERS_D)
        elif BN_D:
            out_dir = '/u/grewalka/celeba/proxy/bn_d/%d_1/' % (ITERS_D)
        elif BN_G:
            out_dir = '/u/grewalka/celeba/proxy/bn_g/%d_1/' % (ITERS_D)
        else:
            out_dir = '/u/grewalka/celeba/proxy/nbn/%d_1/' % (ITERS_D)
    elif METHOD == 'regular':
        if BN_D and BN_G:
            out_dir = '/u/grewalka/celeba/regular/%d_1/' % (ITERS_D)
        elif BN_D:
            out_dir = '/u/grewalka/celeba/regular/bn_d/%d_1/' % (ITERS_D)
        elif BN_G:
            out_dir = '/u/grewalka/celeba/regular/bn_g/%d_1/' % (ITERS_D)
        else:
            out_dir = '/u/grewalka/celeba/regular/nbn/%d_1/' % (ITERS_D)
    else:
        exit(0)

    with open(os.path.join(out_dir, 'out.log'), 'w+') as f:
        f.write('Regular/Proxy GAN\n')
        f.write('D iters: {}\n'.format(ITERS_D))
        f.write('BN on D: {}\tBN on G: {}'.format(BN_D, BN_G))

    X = T.tensor4()
    z = T.fmatrix()

    D, G = discriminator(X, use_batch_norm=BN_D), generator(z, use_batch_norm=BN_G)

    y_real = get_output(D)
    X_fake = get_output(G)
    y_fake = get_output(D, X_fake)
    
    # Regular GAN loss
    D_loss = (y_fake + T.nnet.softplus(-y_real) + T.nnet.softplus(-y_fake)).mean()
    if METHOD == 'regular':
        G_loss = (-(y_fake + T.nnet.softplus(-y_fake))).mean()
    else:
        G_loss = (T.nnet.softplus(-y_fake)).mean() # -log D trick
    
    updates_D = adam(loss_or_grads=D_loss,params=get_all_params(D, trainable=True),
        learning_rate=ADAM_LR,beta1=ADAM_BETA1,beta2=ADAM_BETA2,epsilon=ADAM_EPSILON)

    updates_G = adam(loss_or_grads=G_loss,params=get_all_params(G, trainable=True),
        learning_rate=ADAM_LR,beta1=ADAM_BETA1,beta2=ADAM_BETA2,epsilon=ADAM_EPSILON)

    train_D = function([X, z], outputs=D_loss, updates=updates_D, allow_input_downcast=True)
    train_G = function([z], outputs=G_loss, updates=updates_G, allow_input_downcast=True)

    # Gradient Norms
    G_loss_grad = T.grad(G_loss, X_fake)
    G_loss_grad_norm_value = function([z],outputs=(G_loss_grad**2).sum(axis=(1,2,3)).mean())
    D_loss_grad = T.grad(D_loss, X_fake)
    D_loss_grad_norm_value = function([X, z],outputs=(D_loss_grad**2).sum(axis=(1,2,3)).mean())

    # Sampling functions
    generate = function([z], outputs=X_fake)
    D_out_R = function([X], outputs=y_real)
    D_out_F = function([z], outputs=y_fake)

    print('Output files will be placed in: {}'.format(out_dir))

    # Fixed z for generating samples
    z_samples = np.float32(np.random.normal(size=(BATCH_SIZE, DIM_Z)))

    # Load data
    stream, num_examples = get_data(BATCH_SIZE)

    for epoch in range(EPOCHS):
        print('Starting Epoch {}/{} ...'.format(epoch+1, EPOCHS))

        G_loss_grad_norms = np.zeros(shape=(num_examples / BATCH_SIZE))
        D_loss_grad_norms = np.zeros(shape=(num_examples / BATCH_SIZE, ITERS_D))
        
        # D samples
        D_samples = np.zeros(shape=((num_examples / BATCH_SIZE)*BATCH_SIZE , 2))

        for i in range(num_examples / BATCH_SIZE):

            # Train Discriminator
            iterator = stream.get_epoch_iterator()
            for k in range(ITERS_D):
                x_i = iterator.next()[0]
                z_i = np.float32(np.random.normal(size=(BATCH_SIZE, DIM_Z)))
                train_D(x_i, z_i)
                D_loss_grad_norms[i,k] = D_loss_grad_norm_value(x_i, z_i)

            # train generator
            z_i = np.float32(np.random.normal(size=(BATCH_SIZE, DIM_Z)))
            train_G(z_i)
            G_loss_grad_norms[i] = G_loss_grad_norm_value(z_i)

        # Generate samples from G
        x_samples = generate(z_samples)
        with open(os.path.join(out_dir, 'x_samples_%d.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, x_samples)

        # Sample from D
        iterator = stream.get_epoch_iterator()
        for i in range(num_examples / BATCH_SIZE):
            x_i = iterator.next()[0]
            z_i = np.float32(np.random.normal(size=(BATCH_SIZE, DIM_Z)))
            D_samples[i*BATCH_SIZE:(i+1)*BATCH_SIZE,0:1] = D_out_R(x_i)
            D_samples[i*BATCH_SIZE:(i+1)*BATCH_SIZE,1:2] = D_out_F(z_i)

        # Save Results
        with open(os.path.join(out_dir, 'D_samples_%d.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, D_samples)
        
        # Save model
        if epoch % 5 == 4:
            D_params = get_all_params(get_all_layers(D))
            with open(os.path.join(out_dir, 'discriminator_model_{}.npz'.format(epoch+1)), 'w+') as f:
                np.savez(f, D_params)
            
            G_params = get_all_params(get_all_layers(G))
            with open(os.path.join(out_dir, 'generator_model_{}.npz'.format(epoch+1)), 'w+') as f:
                np.savez(f, G_params)
        
        # Save gradient norms
        with open(os.path.join(out_dir, 'D_loss_grad_norms_%d.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, D_loss_grad_norms)
        with open(os.path.join(out_dir, 'G_loss_grad_norms_%d.npz' % (epoch+1)), 'w+') as f:
            np.savez(f, G_loss_grad_norms)



