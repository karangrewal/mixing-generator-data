import numpy as np

from lasagne.layers import get_all_params, get_output
from lasagne.updates import adam, sgd

from theano import function
import theano.tensor as T

from data import get_data
from model import discriminator, generator

if __name__ == '__main__':
    # place params in separate file
    params = {
        'adam_beta1':0.9,
        'adam_beta2':0.999,
        'adam_epsilon':3e-6,
        'adam_learning_rate':0.005,
        'batch_size':32,
        'discriminator_iters':1,
        'epochs':1,
        'gamma':0.1
    }

    X = T.tensor4()
    z_f = T.fmatrix()
    z = T.fmatrix()

    D, G = discriminator(X), generator(z)

    y_real = get_output(D)
    X_fake = get_output(G)
    X_fake_flip = get_output(G, inputs=z_f)
    y_fake = get_output(D, inputs=X_fake)
    y_fake_flip = get_output(D, inputs=X_fake_flip)

    # Discriminator Loss
    # -E_{x~p_d} [log D(x)] - E_{z'~p_z} [log D(G(z))] - E_{z~p_z} [log(1-D(G(z)))]
    # where z' are the samples whose labels will be flipped
    D_loss = T.nnet.softplus(-y_real).mean() + T.nnet.softplus(-y_fake_flip).mean() + y_fake.mean() + T.nnet.softplus(-y_fake).mean()
    
    # Generator Loss
    # E_{z~p_z} [log(1-D(G(z)))]
    G_loss = -(y_fake.mean() + T.nnet.softplus(-y_fake).mean())

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

    train_D = function(
        inputs=[X, z_f, z],
        outputs=D_loss,
        updates=updates_D,
        allow_input_downcast=True
    )

    train_G = function(
        inputs=[z],
        outputs=G_loss,
        updates=updates_G,
        allow_input_downcast=True
    )

    # Value of discriminator
    D_value = function([z], y_fake)

    # Value of loss functions
    D_loss_value = function(inputs=[X, z_f, z],outputs=D_loss)
    G_loss_value = function(inputs=[z],outputs=G_loss)

    D_grad_norm = (T.grad(D_loss, X) ** 2).sum(axis=(1,2,3)).mean()
    G_grad_norm = (T.grad(G_loss, X_fake) ** 2).sum(axis=(1,2,3)).mean()

    # Value of E||grad(dL/dx)||^2
    D_grad_norm_value = function(inputs=[X, z_f, z],outputs=D_grad_norm)
    G_grad_norm_value = function(inputs=[z],outputs=G_grad_norm)

    # Load data
    batches = get_data(params['batch_size'])

    print('Starting Training. params['gamma'] = {}'.format(params['gamma']))
    for epoch in range(params['epochs']):
        print('\nStarting Epoch {}/{} ...\n'.format(epoch+1, params['epochs']))
        D_losses = np.zeros(shape=(batches.shape[0], params['discriminator_iters']))
        G_losses = np.zeros(shape=(batches.shape[0]))
        D_grad_norms = np.zeros(shape=(batches.shape[0], params['discriminator_iters']))
        G_grad_norms = np.zeros(shape=(batches.shape[0]))

        # Train Discriminator
        for k in range(params['discriminator_iters']):
            print('\tDisc. Iter: {}/{}'.format(k+1, params['discriminator_iters']))
            
            # Sample from data distribution and train discriminator
            np.random.shuffle(batches)
            for i in range(batches.shape[0]):
                x_i = batches[i]
                z_i = np.float32(np.random.normal(size=(params['batch_size'],100)))# CHANGE TO DIMZ
                
                # Need to Evaluate best z_i's
                scores = D_value(z_i)
                scores = scores.reshape(scores.shape[0])
                indices = np.argsort(scores)[::-1][:int(round(params['gamma']*scores.shape[0]))]
                z_f_i = z_i[indices]
                if z_f_i.shape[0] == 0:
                    z_f_i = np.float32(np.zeros(shape=(1,100)))
                z_i = np.delete(z_i, indices, axis=0)

                if i % 600 == 0:
                    print('\t\tbatch: {}/{}\n\t\tDisc. Loss: {}'.format(i, batches.shape[0], D_loss_value(x_i, z_f_i, z_i)))
                    print('\t\tD_grad_norm: {}'.format(D_grad_norm_value(x_i, z_f_i, z_i)))
                train_D(x_i, z_f_i, z_i)
                D_losses[i,k] = D_loss_value(x_i, z_f_i, z_i)
                D_grad_norms[i,k] = D_grad_norm_value(x_i, z_f_i, z_i)

        # train generator
        for i in range(batches.shape[0]):
            z_i = np.float32(np.random.normal(size=(params['batch_size'],100)))# CHANGE TO DIMZ
            scores = D_value(z_i)
            scores = scores.reshape(scores.shape[0])
            indices = np.argsort(scores)[::-1][:int(round(params['gamma']*scores.shape[0]))]
            z_f_i = z_i[indices]
            if z_f_i.shape[0] == 0:
                z_f_i = np.float32(np.zeros(shape=(1,100)))
            z_i = np.delete(z_i, indices, axis=0)

            train_G(z_i)
            G_losses[i] = G_loss_value(z_i)
            G_grad_norms[i] = G_grad_norm_value(z_i)

        # End of epoch
        with open('D_losses_%d.npz' % epoch, 'w+') as f:
            np.savez(f, D_losses, delimiter=',')
        with open('D_grad_norms_%d.npz' % epoch, 'w+') as f:
            np.savez(f, D_grad_norms, delimiter=',')
        with open('G_losses_%d.npz' % epoch, 'w+') as f:
            np.savez(f, G_losses, delimiter=',')
        with open('G_grad_norms_%d.npz' % epoch, 'w+') as f:
            np.savez(f, G_grad_norms, delimiter=',')
