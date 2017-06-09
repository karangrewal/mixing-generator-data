"""
Increasing variance in the discriminator output of GANs by forcing it's 
marginal distributions D(X|x~p) and D(X|x~q) to be gaussian with means 1 and 0 
respectively.
"""

import numpy as np
from lasagne.layers import get_all_params, get_output, get_all_layers
from lasagne.objectives import binary_crossentropy
from lasagne.updates import adam, sgd
from theano import function
import theano.tensor as T

from data import get_data
from model import discriminator, generator
from real_fake import real_fake_discriminator

if __name__ == '__main__':
    # place params in separate file
    params = {
        'adam_beta1':0.9,
        'adam_beta2':0.999,
        'adam_epsilon':3e-6,
        'adam_learning_rate':0.005,
        'batch_size':32,
        'dim_z':100,
        'iters_D':1,
        'iters_F':1,
        'iters_R':1,
        'epochs':1
    }

    X = T.tensor4()
    z = T.fmatrix()

    D, G = discriminator(X), generator(z)

    y_real = get_output(D)
    X_fake = get_output(G)
    y_fake = get_output(D, inputs=X_fake)

    # Real and fake discriminators
    F, R = real_fake_discriminator(y_fake), real_fake_discriminator(y_real)

    # Temporary names
    r_real = T.fmatrix()
    r_fake = get_output(R)
    f_real = T.fmatrix()
    f_fake = get_output(F)

    # Loss functions
    F_loss = (T.nnet.softplus(-f_real) + T.nnet.softplus(-f_fake) + f_fake).mean()
    R_loss = (T.nnet.softplus(-r_real) + T.nnet.softplus(-r_fake) + r_fake).mean()
    D_loss = (T.nnet.softplus(-f_fake) + T.nnet.softplus(-r_fake)).mean()
    '''
    F_loss = -(T.log(1.-f_fake) + T.log(f_real)).mean()
    R_loss = -(T.log(1.-r_fake) + T.log(r_real)).mean()
    D_loss = -(T.log(f_fake) + T.log(r_fake)).mean()
    '''
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

    train_F = function(inputs=[f_real, z], outputs=F_loss, updates=updates_F, allow_input_downcast=True)
    train_R = function(inputs=[r_real, X], outputs=R_loss, updates=updates_R, allow_input_downcast=True)
    train_D = function(inputs=[X, z], outputs=D_loss, updates=updates_D, allow_input_downcast=True)
    train_G = function(inputs=[z], outputs=G_loss, updates=updates_G, allow_input_downcast=True)

    # Gradient Norms
    F_grad_norm = (T.grad(F_loss, X_fake) ** 2).sum(axis=(0,1,2,3))
    R_grad_norm = (T.grad(R_loss, X) ** 2).sum(axis=(0,1,2,3))
    D_grad_norm = (T.grad(D_loss, X) ** 2).sum(axis=(0,1,2,3))
    G_grad_norm = (T.grad(G_loss, X_fake) ** 2).sum(axis=(0,1,2,3))

    # Value of E||grad(dL/dx)||^2
    F_grad_norm_value = function(inputs=[f_real, z],outputs=F_grad_norm)
    R_grad_norm_value = function(inputs=[r_real, X],outputs=R_grad_norm)
    D_grad_norm_value = function(inputs=[X, z],outputs=D_grad_norm)
    G_grad_norm_value = function(inputs=[z],outputs=G_grad_norm)

    # Load data
    batches = get_data(params['batch_size'])

    for epoch in range(params['epochs']):
        print('\nStarting Epoch {}/{} ...\n'.format(epoch+1, params['epochs']))

        # Keep track of loss values
        F_losses = np.zeros(shape=(batches.shape[0], params['iters_F']))
        R_losses = np.zeros(shape=(batches.shape[0], params['iters_R']))
        D_losses = np.zeros(shape=(batches.shape[0], params['iters_D']))
        G_losses = np.zeros(shape=(batches.shape[0]))

        # Keep track of gradient norms
        F_grad_norms = np.zeros(shape=(batches.shape[0], params['iters_F']))
        R_grad_norms = np.zeros(shape=(batches.shape[0], params['iters_R']))
        D_grad_norms = np.zeros(shape=(batches.shape[0], params['iters_D']))
        G_grad_norms = np.zeros(shape=(batches.shape[0]))

        # Train fake data discriminator
        for k in range(params['iters_F']):
            for i in range(batches.shape[0]):
                f_real_i = np.float32(np.random.normal(size=(params['batch_size'],1)))
                z_i = np.float32(np.random.normal(size=(params['batch_size'],params['dim_z'])))
                F_losses[i,k] = train_F(f_real_i, z_i)
                F_grad_norms[i,k] = F_grad_norm_value(f_real_i, z_i)

        # Train real data discriminator
        for k in range(params['iters_R']):
            np.random.shuffle(batches)
            for i in range(batches.shape[0]):
                r_real_i = np.float32(np.random.normal(loc=1.0,size=(params['batch_size'],1)))
                x_i = batches[i]
                R_losses[i,k] = train_R(r_real_i, x_i)
                R_grad_norms[i,k] = R_grad_norm_value(r_real_i, x_i)

        # Train discriminator
        for k in range(params['iters_D']):
            np.random.shuffle(batches)
            for i in range(batches.shape[0]):
                x_i = batches[i]
                z_i = np.float32(np.random.normal(size=(params['batch_size'],params['dim_z'])))
                D_losses[i,k] = train_D(x_i, z_i)
                D_grad_norms[i,k] = D_grad_norm_value(x_i, z_i)

        # train generator
        for i in range(batches.shape[0]):
            z_i = np.float32(np.random.normal(size=(params['batch_size'],params['dim_z'])))
            G_losses[i] = train_G(z_i)
            G_grad_norms[i] = G_grad_norm_value(z_i)

        # End of epoch
        with open('/u/grewalka/lasagne/F_losses.npz', 'w+') as f:
            np.savez(f, F_losses, delimiter=',')
        with open('/u/grewalka/lasagne/F_grad_norms.npz', 'w+') as f:
            np.savez(f, F_grad_norms, delimiter=',')
        with open('/u/grewalka/lasagne/R_losses.npz', 'w+') as f:
            np.savez(f, R_losses, delimiter=',')
        with open('/u/grewalka/lasagne/R_grad_norms.npz', 'w+') as f:
            np.savez(f, R_grad_norms, delimiter=',')
        with open('/u/grewalka/lasagne/D_losses.npz', 'w+') as f:
            np.savez(f, D_losses, delimiter=',')
        with open('/u/grewalka/lasagne/D_grad_norms.npz', 'w+') as f:
            np.savez(f, D_grad_norms, delimiter=',')
        with open('/u/grewalka/lasagne/G_losses.npz', 'w+') as f:
            np.savez(f, G_losses, delimiter=',')
        with open('/u/grewalka/lasagne/G_grad_norms.npz', 'w+') as f:
            np.savez(f, G_grad_norms, delimiter=',')
