#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 13:08:21 2025

@author: lavanyasanapala
"""

# -*- coding: utf-8 -*-
"""
MNIST : CNN

Attack types: CW, PGD and FGSM

Pairwise kenel deviations for the identification of istributionl shofts by Adversarial examples

Non-parametric NUll Hypothesis Evaluation framework


"""

#pip install tensorflow



import cleverhans

import numpy as np
import tensorflow as tf
import keras
#import tensorflow_datasets as tfds
from matplotlib import pyplot as plt
import sklearn
from sklearn.metrics import confusion_matrix, r2_score, mean_squared_error
import numpy as np

import matplotlib.pyplot as plt

from numpy import asarray
from numpy import unique
from numpy import argmax
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras import Sequential

# Commented out IPython magic to ensure Python compatibility.
# example of loading and plotting the mnist dataset
from tensorflow.keras.datasets.mnist import load_data
from matplotlib import pyplot

from sklearn.metrics import classification_report


# %matplotlib inline
# load dataset
(trainX, trainy), (testX, testy) = load_data()
# summarize loaded dataset
print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))
print('Test: X=%s, y=%s' % (testX.shape, testy.shape))
# plot first few images
for i in range(25):
	# define subplot
	pyplot.subplot(5, 5, i+1)
	# plot raw pixel data
	pyplot.imshow(trainX[i], cmap=pyplot.get_cmap('gray'))
# show the figure
pyplot.show()

trainX.ndim

# load dataset
(x_train, y_train), (x_test, y_test) = load_data()
# reshape data to have a single channel
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))
# determine the shape of the input images
in_shape = x_train.shape[1:]
# determine the number of classes
n_classes = len(unique(y_train))
print(in_shape, n_classes)
# normalize pixel values
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0


# plot first few images
for i in range(25):
	# define subplot
	pyplot.subplot(5, 5, i+1)
	# plot raw pixel data
	pyplot.imshow(x_train[i], cmap=pyplot.get_cmap('gray'))
# show the figure
pyplot.show()

# define model
# CNN for MNIST training


model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', kernel_initializer='he_uniform', input_shape=in_shape))
model.add(Conv2D(64, (3,3), activation='relu', kernel_initializer='he_uniform'))
model.add(Conv2D(128,(3,3), activation='relu', kernel_initializer='he_uniform'))
model.add(MaxPool2D((2, 2)))
model.add(Flatten())
model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
model.add(Dropout(0.5))
model.add(Dense(n_classes, activation='softmax'))
# define loss and optimizer
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# fit the model
model.fit(x_train, y_train, epochs=15, batch_size=128, verbose=1)

model.summary()

#Create model checkpoint and save it for later usage

Model_path = "mnist_dist_det_DCNN.pth"


# evaluate the model
loss, acc = model.evaluate(x_test, y_test, verbose=1)
print('CNN clean Model Accuracy on MNIST : %.4f' % (acc*100))
# make a prediction
image = x_train[1]
yhat = model.predict(asarray([image]))
print('Predicted: class=%d' % argmax(yhat))

plt.imshow(x_train[1])

# CNN prediction score
y_pred_NN = model.predict(x_test)
y_score = acc
y_score



# your model predictions probabilities in `y_pred_proba`
y_pred_proba = model.predict(x_test)

# Convert predicted probabilities to class labels
y_pred = np.argmax(y_pred_proba, axis=1)

# Then print the classification report:
print(classification_report(y_test, y_pred))

# Perform cleverhans attacks on MNIST dataset

from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method

from cleverhans.tf2.utils import optimize_linear, compute_gradient

import sklearn
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics.pairwise import rbf_kernel, laplacian_kernel

# Fsgm attack 1% = 600, 5% = 3000, 10% = 6000 eps = {3e-1, 1.5e-2, 2e-3, 1, 2.7, and 3}
n_tr = 60000
n_pois = 3000
x=x_train[:n_pois, :]
y = y_test[:n_pois]
fgm = fast_gradient_method(model,
    x,
    eps=3e-1,
    norm=2,
    clip_min=0.,
    clip_max=1.,
    y=None,
    targeted=False,
    sanity_checks=False,
)

fgm.shape

adv_x = fgm

fgm

for i in range(25):
# define subplot
  plt.subplot(5, 5, i+1)
	# plot raw pixel data
  plt.imshow(np.resize(adv_x[i],(28,28)), cmap=plt.get_cmap('gray'))
# show the figure
#plt.subplot.title('Poisoned Samples')
plt.show()

los, pois_acc = model.evaluate(adv_x, y, verbose=1)
print('NN Model Accuracy on MNIST after Attack: %.3f' % (pois_acc*100))
print('No.of Poisoning points : ', n_pois)

y_pred_fgm = model(adv_x)
y_pred_adv = model.predict(adv_x)
y_pred_adv

yt = y_test[:n_pois]

# your model predictions probabilities in `y_pred_proba`
y_pred_proba_fgm = model.predict(adv_x)

# Convert predicted probabilities to class labels
y_pred_adv_fgm = np.argmax(y_pred_proba_fgm, axis=1)

# Then print the classification report:
print(classification_report(yt, y_pred_adv_fgm))




test_predictions = model.predict(x_test)
mnist_conf = confusion_matrix(y_test, np.argmax(test_predictions,axis=1))

mnist_conf

plt.imshow(mnist_conf, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('MNIST CNN Confusion matrix')
plt.colorbar()

classes = [str(i) for i in range(10)]
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()

y_tf = y_test[:n_pois]
test_predictions_fgm = model.predict(fgm)
pois_model_confusion_fgm = confusion_matrix(y_tf, np.argmax(test_predictions_fgm,axis=1))

pois_model_confusion_fgm

plt.imshow(pois_model_confusion_fgm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('FGSM Confusion matrix')
plt.colorbar()

classes = [str(i) for i in range(10)]
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()

plt.imshow(np.resize(fgm[7],(28,28)), cmap=plt.get_cmap('gray'))

pyplot.imshow(np.resize(x[7],(28,28)), cmap=pyplot.get_cmap('gray'))

for i in range(20):
  pyplot.subplot(4, 5, i+1)
  pyplot.imshow(np.resize(fgm[i],(28,28)), cmap=pyplot.get_cmap('gray'))
pyplot.show()

c2=[]
for i in range(600):
  c2.append(x[i])
c2

# Pre-processing data for stat-test
c=np.array(x)
c.shape

c = np.resize(c, (28,28))
c.shape

f = np.array(fgm)
f.shape

f= np.resize(f, (28,28))
f.shape



plt.hist(c)
plt.show()

pois_score = y_pred_adv
pois_score

pois_samp = fgm.numpy()
pois_samp.shape


adv_X = adv_x.numpy()

p_s_fgm =  np.resize(adv_X, (28, 28))

p_s_fgm.ndim

p_s_fgm

x.ndim
x
c_s = np.resize(x, (28,28))
c_s
c_s.ndim
c_s.shape

plt.hist(c_s[25])

plt.hist(p_s_fgm[25])

plt.hist(c_s)
plt.hist(p_s_fgm)

print("Clean MNIST samples peak-to-peak Value: ", x_train.ptp())
print("Reduced clean samples (1500) P-2-P value: ", x.ptp())
print("Poisoned Samples peak-to-peak value : ",adv_x.numpy().ptp())

x.mean()



# Distance-based detection using pairwise kernel statistical distance metrics
# Modified RBF function to calculate Deviation

def rbf_deviation(x,y,gamma):
  distance = rbf_kernel(x,y, gamma=gamma)  # Euclidean distance
  r_deviation = 1 - distance
  return r_deviation

# Modified Laplace function to calculate deviation

def Laplace_deviation(x,y,gamma):
  dist = laplacian_kernel(x, y, gamma=gamma)
  deviation = 1-dist
  return deviation

Rbf_d_fgm = rbf_deviation(c_s, p_s_fgm, 1)
print("RBF deviations obtained between clean and FGSM MNIST \n ", Rbf_d_fgm)
print("RBF mean Deviation fgsm poisoning on MNIST : ", Rbf_d_fgm.mean())


plt.title ='RBF measure between clean and fgsm MNIST images'
plt.hist(Rbf_d_fgm, label='RBF measure between clean and fgsm MNIST images')
plt.show()

Rbf_d_clean = rbf_deviation(c_s, c_s,None)
print("RBF mean Deviation clean F_MNIST : ", Rbf_d_clean.mean())


L_d_fgm = Laplace_deviation(c_s, p_s_fgm,1)
print("Laplace Deviation fgsm poisoning on F_MNIST: ", L_d_fgm.mean())

plt.hist(L_d_fgm)

# ----- Deviation Plots -----

#---- RBF plots -----

gamma = 1

# Natural clean-to-clean deviation (shuffle or pairwise samples)
np.random.seed(42)  # reproducibility
idx = np.random.permutation(len(c_s[1]))
c_s_shuffled = c_s[idx]
Rbf_d_clean = rbf_deviation(c_s, c_s, None)

# Summary statistics
print("FGSM deviation: mean={:.3f}, min={:.3f}, max={:.3f}".format(Rbf_d_fgm.mean(), Rbf_d_fgm.min(), Rbf_d_fgm.max()))
print("Clean deviation: mean={:.3f}, min={:.3f}, max={:.3f}".format(Rbf_d_clean.mean(), Rbf_d_clean.min(), Rbf_d_clean.max()))

# ----- Graphical comparison -----
plt.figure(figsize=(10,6))
plt.hist(Rbf_d_clean.flatten(), bins=50, alpha=0.5, label='Clean-to-Clean', color='green', edgecolor='black')
plt.hist(Rbf_d_fgm.flatten(), bins=50, alpha=0.5, label='Clean-to-FGSM', color='skyblue', edgecolor='black')
plt.axvline(Rbf_d_clean.mean(), color='green', linestyle='--', label=f'Clean mean = {Rbf_d_clean.mean():.3f}')
plt.axvline(Rbf_d_fgm.mean(), color='blue', linestyle='--', label=f'FGSM mean = {Rbf_d_fgm.mean():.3f}')
plt.title = "RBF Deviation: Clean vs FGSM-Poisoned MNIST"
plt.xlabel("RBF Deviation (1 - similarity)")
plt.ylabel("Frequency")
plt.legend()
plt.show()



#---- Laplacian plots -----

gamma = 1

# Natural clean-to-clean deviation (shuffle or pairwise samples)
np.random.seed(42)  # reproducibility
idx = np.random.permutation(len(c_s[1]))
c_s_shuffled = c_s[idx]
L_d_clean = Laplace_deviation(c_s, c_s, None)

# Summary statistics
print("FGSM deviation: mean={:.3f}, min={:.3f}, max={:.3f}".format(L_d_fgm.mean(), L_d_fgm.min(), L_d_fgm.max()))
print("Clean deviation: mean={:.3f}, min={:.3f}, max={:.3f}".format(L_d_clean.mean(), L_d_clean.min(), L_d_clean.max()))

# ----- Graphical comparison -----
plt.figure(figsize=(10,6))
plt.hist(L_d_clean.flatten(), bins=50, alpha=0.5, label='Clean-to-Clean', color='green', edgecolor='black')
plt.hist(L_d_fgm.flatten(), bins=50, alpha=0.5, label='Clean-to-FGSM', color='skyblue', edgecolor='black')
plt.axvline(L_d_clean.mean(), color='green', linestyle='--', label=f'Clean mean = {L_d_clean.mean():.3f}')
plt.axvline(L_d_fgm.mean(), color='blue', linestyle='--', label=f'FGSM mean = {L_d_fgm.mean():.3f}')
plt.title = "Laplace Deviation: Clean vs FGSM-Poisoned MNIST"
plt.xlabel("Laplace Deviation (1 - similarity)")
plt.ylabel("Frequency")
plt.legend()
plt.show()






from cleverhans.tf2.utils import clip_eta, random_lp_vector



def projected_gradient_descent(
    model_fn,
    x,
    eps,
    eps_iter,
    nb_iter,
    norm,
    loss_fn=None,
    clip_min=None,
    clip_max=None,
    y=None,
    targeted=False,
    rand_init=None,
    rand_minmax=None,
    sanity_checks=False):
  assert eps_iter <= eps, (eps_iter, eps)
  if norm == 1:
        raise NotImplementedError(
            "It's not clear that FGM is a good inner loop"
            " step for PGD when norm=1, because norm=1 FGM "
            " changes only one pixel at a time. We need "
            " to rigorously test a strong norm=1 PGD "
            "before enabling this feature."
        )
  if norm not in [np.inf, 2]:
        raise ValueError("Norm order must be either np.inf or 2.")

  if loss_fn is None:
        loss_fn = tf.nn.sparse_softmax_cross_entropy_with_logits

  asserts = []

    # If a data range was specified, check that the input was in that range
  if clip_min is not None:
        asserts.append(tf.math.greater_equal(x, clip_min))

  if clip_max is not None:
        asserts.append(tf.math.less_equal(x, clip_max))

    # Initialize loop variables
  if rand_minmax is None:
        rand_minmax = eps

  if rand_init:
        eta = random_lp_vector(
            tf.shape(x), norm, tf.cast(rand_minmax, x.dtype), dtype=x.dtype
        )
  else:
        eta = tf.zeros_like(x)

    # Clip eta
  eta = clip_eta(eta, norm, eps)
  adv_x = x + eta
  if clip_min is not None or clip_max is not None:
        adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)

  if y is None:
        # Using model predictions as ground truth to avoid label leaking
        y = tf.argmax(model_fn(x), 1)

  i = 0
  while i < nb_iter:
        adv_x = fast_gradient_method(
            model_fn,
            adv_x,
            eps_iter,
            norm,
            loss_fn,
            clip_min=clip_min,
            clip_max=clip_max,
            y=y,
            targeted=targeted,
        )

        # Clipping perturbation eta to norm norm ball
        eta = adv_x - x
        eta = clip_eta(eta, norm, eps)
        adv_x = x + eta

        # Redo the clipping.
        # FGM already did it, but subtracting and re-adding eta can add some
        # small numerical error.
        if clip_min is not None or clip_max is not None:
            adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)
        i += 1

  asserts.append(eps_iter <= eps)
  if norm == np.inf and clip_min is not None:
        # TODO necessary to cast to x.dtype?
        asserts.append(eps + clip_min <= clip_max)

  if sanity_checks:
        assert np.all(asserts)
  return adv_x

n_pois = 3000
x=x_train[:n_pois, :]
y = y_test[:n_pois]

y.shape

pois_pgd = projected_gradient_descent(model,
    x,
    eps = 3e-1,
    eps_iter =0.1e-6,
    nb_iter = 10,
    norm =2,
    clip_min=0.,
    clip_max=1.,
    y=None,
    targeted=False,
    rand_init=None,
    rand_minmax=None,
    sanity_checks=False)

#pois_pgd.ndim

pgd_adv = pois_pgd

los, pois_acc = model.evaluate(pois_pgd, y, verbose=1)
print('CNN Model Accuracy on F_MNIST after Attack: %.2f' % (pois_acc*100))
print('No.of Poisoning points : ', n_pois)

y_pred_pgd = model(pois_pgd)

plt.hist(y_pred_pgd)
plt.show()

pois_pgd_samp = pgd_adv.numpy()
pgd_p_s =  np.resize(pois_pgd_samp, (28, 28))
pgd_p_s.ndim

for i in range(25):
# define subplot
  plt.subplot(5, 5, i+1)
	# plot raw pixel data
  plt.imshow(np.resize(pois_pgd_samp[i],(28,28)), cmap=plt.get_cmap('gray'))
# show the figure
plt.title = 'poisoned samples'
plt.show()

pgd_p_s.mean()



Rbf_d_pgd = rbf_deviation(c_s, pgd_p_s, 1)
#print("RBF PGD Deviation  : ", Rbf_d_pgd)
print("RBF PGD Mean Deviation : ", Rbf_d_pgd.mean())

plt.hist(Rbf_d_pgd,histtype = 'step')

L_d_pgd = Laplace_deviation(c_s, pgd_p_s,1)
#print("Laplace PGD Deviation : ", L_d_pgd)
plt.hist(L_d_pgd,histtype = 'step')
print("Laplace PGD mean Deviation : ", L_d_pgd.mean())


#-------- Deviation Plots---------

#---- RBF plots -----

gamma = 1

# Natural clean-to-clean deviation (shuffle or pairwise samples)
np.random.seed(42)  # reproducibility
idx = np.random.permutation(len(c_s[1]))
c_s_shuffled = c_s[idx]
Rbf_d_clean = rbf_deviation(c_s, c_s, None)

# Summary statistics
print("PGD deviation: mean={:.3f}, min={:.3f}, max={:.3f}".format(Rbf_d_pgd.mean(), Rbf_d_pgd.min(), Rbf_d_pgd.max()))
print("Clean deviation: mean={:.3f}, min={:.3f}, max={:.3f}".format(Rbf_d_clean.mean(), Rbf_d_clean.min(), Rbf_d_clean.max()))

# ----- Graphical comparison -----
plt.figure(figsize=(10,6))
plt.hist(Rbf_d_clean.flatten(), bins=50, alpha=0.5, label='Clean-to-Clean', color='green', edgecolor='black')
plt.hist(Rbf_d_pgd.flatten(), bins=50, alpha=0.5, label='Clean-to-PGD', color='skyblue', edgecolor='black')
plt.axvline(Rbf_d_clean.mean(), color='green', linestyle='--', label=f'Clean mean = {Rbf_d_clean.mean():.3f}')
plt.axvline(Rbf_d_pgd.mean(), color='blue', linestyle='--', label=f'PGD mean = {Rbf_d_pgd.mean():.3f}')
plt.title = "RBF Deviation: Clean vs PGD-Poisoned MNIST"
plt.xlabel("RBF Deviation (1 - similarity)")
plt.ylabel("Frequency")
plt.legend()
plt.show()



#---- Laplacian plots -----

gamma = 1

# Natural clean-to-clean deviation (shuffle or pairwise samples)
np.random.seed(42)  # reproducibility
idx = np.random.permutation(len(c_s[1]))
c_s_shuffled = c_s[idx]
L_d_clean = Laplace_deviation(c_s, c_s, None)

# Summary statistics
print("PGD deviation: mean={:.3f}, min={:.3f}, max={:.3f}".format(L_d_pgd.mean(), L_d_pgd.min(), L_d_pgd.max()))
print("Clean deviation: mean={:.3f}, min={:.3f}, max={:.3f}".format(L_d_clean.mean(), L_d_clean.min(), L_d_clean.max()))

# ----- Graphical comparison -----
plt.figure(figsize=(10,6))
plt.hist(L_d_clean.flatten(), bins=50, alpha=0.5, label='Clean-to-Clean', color='green', edgecolor='black')
plt.hist(L_d_pgd.flatten(), bins=50, alpha=0.5, label='Clean-to-PGD', color='skyblue', edgecolor='black')
plt.axvline(L_d_clean.mean(), color='green', linestyle='--', label=f'Clean mean = {L_d_clean.mean():.3f}')
plt.axvline(L_d_pgd.mean(), color='blue', linestyle='--', label=f'PGD mean = {L_d_pgd.mean():.3f}')
plt.title = "Laplace Deviation: Clean vs PGD-Poisoned MNIST"
plt.xlabel("Laplace Deviation (1 - similarity)")
plt.ylabel("Frequency")
plt.legend()
plt.show()



#-------- Carlini and Wagner Attack and Deviation measures --------------


from cleverhans.tf2.utils import get_or_guess_labels, set_with_mask

eps = 3e-1

# Carlini-Wagner L2 (untargeted) - simplified implementation
# This implementation performs per-batch optimization of a perturbation variable w where
# x_adv = 0.5*(tanh(w)+1) mapped to original input range. Because inputs are preprocessed
# using VGG preprocess_input (which subtracts mean), we will work in the same preprocessed space
# and perform box constraint clipping. This is a simplified and slower version; for production use
# consider using established libraries (faster, more options).

def cw_l2_attack(model, x_np, y_np, targeted=False, target_labels=None,
                      c=eps, kappa=0, maxiter=100, lr=1e-2):
    """
    x_np: numpy array batch of inputs (preprocessed)
    y_np: numpy integer labels
    Returns adversarial examples (numpy)
    """
    batch_size = x_np.shape[0]
    x_var = tf.convert_to_tensor(x_np)
    # Initialize w such that tanh(w) ~= (x) mapped to (-1,1)
    # We'll map preprocessed inputs into a roughly bounded range before inverse tanh.

    # safe epsilon to avoid atanh issues
    eps_atanh = 1e-6

    # map x to [-1,1] approx by dividing by a heuristic range
    # For VGG preprocess_input, values roughly lie in [-123.68, 151.061] per channel after preprocessing.
    # Choose scale S to cover this range
    S = 200.0
    x_scaled = x_np / S
    x_scaled = np.clip(x_scaled, -0.999999, 0.999999)
    # inverse tanh
    w_init = 0.5 * np.log((1 + x_scaled) / (1 - x_scaled + 1e-12))

    w = tf.Variable(w_init, dtype=tf.float32)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    y_true = tf.convert_to_tensor(y_np)

    targeted = bool(targeted)
    if targeted and target_labels is None:
        raise ValueError('targeted attack requires target_labels')

    for step in range(maxiter):
        with tf.GradientTape() as tape:
            # generate x_adv from w
            x_adv = tf.tanh(w)  # in (-1,1)
            x_adv = x_adv * S  # scale back to preprocessed range

            logits = model(x_adv, training=False)

            # Carlini loss: encourage wrong class (untargeted) or target class (targeted)
            real = tf.reduce_sum(tf.one_hot(y_true, 10) * logits, axis=1)
            other = tf.reduce_max((1 - tf.one_hot(y_true, 10)) * logits - (tf.one_hot(y_true, 10) * 10000), axis=1)

            if targeted:
                # want target class score higher than others by margin kappa
                t = tf.reduce_sum(tf.one_hot(target_labels, 10) * logits, axis=1)
                others_max = tf.reduce_max((1 - tf.one_hot(target_labels, 10)) * logits - (tf.one_hot(target_labels, 10) * 10000), axis=1)
                f = tf.maximum(others_max - t + kappa, 0.0)
            else:
                f = tf.maximum(real - other + kappa, 0.0)

            l2dist = tf.reduce_sum(tf.square(x_adv - x_var), axis=[1, 2, 3])
            loss1 = tf.reduce_sum(l2dist)
            loss2 = tf.reduce_sum(c * f)
            loss = loss1 + loss2

        grads = tape.gradient(loss, w)
        optimizer.apply_gradients([(grads, w)])

        if step % 100 == 0:
            # optional logging
            tf.print('CW step', step, 'loss', loss)

        # early stopping when every example is adversarial
        if step % 10 == 0:
            x_adv_np = (np.tanh(w.numpy()) * S).astype(np.float32)
            preds = np.argmax(model.predict(x_adv_np), axis=1)
            if targeted:
                success = (preds == np.array(target_labels)).all()
            else:
                success = (preds != np.array(y_np)).all()
            if success:
                tf.print('CW early stop at step', step)
                break

    x_adv_final = (np.tanh(w.numpy()) * S).astype(np.float32)
    return x_adv_final

n_pois = 3000
x= x_train[:n_pois, :]
y = y_test[:n_pois]

x.shape

#y = np.resize(y, (500,28*28))
#y.shape

c = cw_l2_attack(model, x, y)

CW_Pois = c
CW_Pois

cw_adv = CW_Pois

cw_adv.ndim

los, pois_acc = model.evaluate(CW_Pois, y, verbose=1)
print('CNN Model Accuracy on MNIST after Attack: %.3f' % (pois_acc*100))
print('No.of Poisoning points : ', n_pois)


for i in range(25):
# define subplot
  plt.subplot(5, 5, i+1)
	# plot raw pixel data
  plt.imshow(np.resize(cw_adv[i],(28,28)), cmap=plt.get_cmap('gray'))
# show the figure
plt.title = 'poisoned samples'
plt.show()

#plt.hist(y_pred_cw)
#plt.show()

#pois_CW_samp = cw_adv.numpy()
cw_p_s =  np.resize(cw_adv, (28, 28))
cw_p_s.ndim

# CW perturbation Deviation

Rbf_d_cw = rbf_deviation(c_s, cw_p_s, 1)
#print("RBF deviations for CW poisoned MNIST: ", Rbf_d_cw)
print("RBF CW mean Deviation  : ", Rbf_d_cw.mean())


plt.hist(Rbf_d_cw,histtype = 'step')

L_d_cw = Laplace_deviation(c_s, cw_p_s,1)
#print("Laplacian deviations for CW poisoned MNIST:", L_d_cw)
print("Laplace CW mean Deviation : ", L_d_cw.mean())


# ----- Deviations plots -----

# RBF mean, min, max for clean and FGSM visualizations

gamma = 1


# Natural clean-to-clean deviation (shuffle or pairwise samples)
np.random.seed(42)  # reproducibility
idx = np.random.permutation(len(c_s[1]))
c_s_shuffled = c_s[idx]
Rbf_d_clean = rbf_deviation(c_s, c_s, None)

# Summary statistics
print("CW deviation: mean={:.3f}, min={:.3f}, max={:.3f}".format(Rbf_d_cw.mean(), Rbf_d_cw.min(), Rbf_d_cw.max()))
print("Clean deviation: mean={:.3f}, min={:.3f}, max={:.3f}".format(Rbf_d_clean.mean(), Rbf_d_clean.min(), Rbf_d_clean.max()))

# ----- Graphical comparison -----
plt.figure(figsize=(10,6))
plt.hist(Rbf_d_clean.flatten(), bins=50, alpha=0.5, label='Clean-to-Clean', color='green', edgecolor='black')
plt.hist(Rbf_d_cw.flatten(), bins=50, alpha=0.5, label='Clean-to-CW', color='skyblue', edgecolor='black')
plt.axvline(Rbf_d_clean.mean(), color='green', linestyle='--', label=f'Clean mean = {Rbf_d_clean.mean():.3f}')
plt.axvline(Rbf_d_cw.mean(), color='blue', linestyle='--', label=f'CW mean = {Rbf_d_cw.mean():.3f}')
plt.title = "RBF Deviation: Clean vs CW-Poisoned MNIST"
plt.xlabel("RBF Deviation (1 - similarity)")
plt.ylabel("Frequency")
plt.legend()
plt.show()



# Laplacian mean, min, max for clean and FGSM visualizations

gamma = 1
# PGD perturbation deviation

# Natural clean-to-clean deviation (shuffle or pairwise samples)
np.random.seed(42)  # reproducibility
idx = np.random.permutation(len(c_s[1]))
c_s_shuffled = c_s[idx]
L_d_clean = Laplace_deviation(c_s, c_s, None)

# Summary statistics
print("CW deviation: mean={:.3f}, min={:.3f}, max={:.3f}".format(L_d_cw.mean(), L_d_cw.min(), L_d_cw.max()))
print("Clean deviation: mean={:.3f}, min={:.3f}, max={:.3f}".format(L_d_clean.mean(), L_d_clean.min(), L_d_clean.max()))

# ----- Graphical comparison -----
plt.figure(figsize=(10,6))
plt.hist(L_d_clean.flatten(), bins=50, alpha=0.5, label='Clean-to-Clean', color='green', edgecolor='black')
plt.hist(L_d_cw.flatten(), bins=50, alpha=0.5, label='Clean-to-CW', color='skyblue', edgecolor='black')
plt.axvline(L_d_clean.mean(), color='green', linestyle='--', label=f'Clean mean = {L_d_clean.mean():.3f}')
plt.axvline(L_d_cw.mean(), color='blue', linestyle='--', label=f'CW mean = {L_d_cw.mean():.3f}')
plt.title = "Laplace Deviation: Clean vs CW-Poisoned MNIST"
plt.xlabel("Laplace Deviation (1 - similarity)")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# -------------------------------
# Test statistic: mean deviation
# -------------------------------


x1 = x_train[3000:6000,:]
plt.imshow(x1[60])
x1.shape
x1 = np.resize(x1, (32,32))
plt.imshow(np.resize(x1[10], (32,32)))
x1.shape

Rbf_d_clean = rbf_deviation(x1, x1, None)
L_d_clean = Laplace_deviation(x1, x1, None)

rbf_clean = Rbf_d_clean.flatten()
lap_clean = L_d_clean.flatten()

rbf_fgm = Rbf_d_fgm.flatten()
rbf_pgd = Rbf_d_pgd.flatten()
rbf_cw  = Rbf_d_cw.flatten()

lap_fgm = L_d_fgm.flatten()
lap_pgd = L_d_pgd.flatten()
lap_cw  = L_d_cw.flatten()

#--------- Mann–Whitney U Test (Primary Test)------------

from scipy.stats import mannwhitneyu

def mann_whitney_test(clean, adv, name):
    stat, p = mannwhitneyu(clean, adv, alternative='two-sided')
    print(f"{name}: U-stat={stat:.2e}, p-value={p:.3e}")
    return stat, p


print("\n=== mann_whitneyU_test RBF ===")
mann_whitney_test(rbf_clean, rbf_fgm, "RBF Clean vs FGSM")
mann_whitney_test(rbf_clean, rbf_pgd, "RBF Clean vs PGD")
mann_whitney_test(rbf_clean, rbf_cw,  "RBF Clean vs CW")

print("\n=== mann_whitneyU_test Laplacian===")
mann_whitney_test(lap_clean, lap_fgm, "Laplace Clean vs FGSM")
mann_whitney_test(lap_clean, lap_pgd, "Laplace Clean vs PGD")
mann_whitney_test(lap_clean, lap_cw,  "Laplace Clean vs CW")



#----------Kolmogorov–Smirnov Test (Distribution Shift Proof)------------


from scipy.stats import ks_2samp

def ks_test(clean, adv, name):
    stat, p = ks_2samp(clean, adv)
    print(f"{name}: KS-stat={stat:.3f}, p-value={p:.3e}")
    return stat, p

print("\n=== KS Test (RBF) ===")
ks_test(rbf_clean, rbf_fgm, "RBF Clean vs FGSM")
ks_test(rbf_clean, rbf_pgd, "RBF Clean vs PGD")
ks_test(rbf_clean, rbf_cw,  "RBF Clean vs CW")

print("\n=== KS Test (Laplace) ===")
ks_test(lap_clean, lap_fgm, "Laplace Clean vs FGSM")
ks_test(lap_clean, lap_pgd, "Laplace Clean vs PGD")
ks_test(lap_clean, lap_cw,  "Laplace Clean vs CW")

#--------- Permutation Test (Strongest Evidence)-------------


def permutation_test(clean, adv, n_perm=1000):
    observed = np.abs(clean.mean() - adv.mean())
    combined = np.concatenate([clean, adv])
    count = 0

    for _ in range(n_perm):
        np.random.shuffle(combined)
        new_clean = combined[:len(clean)]
        new_adv   = combined[len(clean):]
        if abs(new_clean.mean() - new_adv.mean()) >= observed:
            count += 1

    return count / n_perm

print("\n=== Permutation test (RBF kernel distance)====")

p_perm_rbf_fgm = permutation_test(rbf_clean, rbf_fgm)
p_perm_rbf_pgd = permutation_test(rbf_clean, rbf_pgd)
p_perm_rbf_cw = permutation_test(rbf_clean, rbf_cw)

print("Permutation p-value (RBF FGSM):", p_perm_rbf_fgm)
print("Permutation p-value (RBF PGD):", p_perm_rbf_pgd)
print("Permutation p-value (RBF CW):", p_perm_rbf_cw)




print("\n=== Permutation test (Laplacian kernel distance)====")
p_perm_lap_fgm = permutation_test(lap_clean, lap_fgm)
p_perm_lap_pgd = permutation_test(lap_clean, lap_pgd)
p_perm_lap_cw = permutation_test(lap_clean, lap_cw)

print("Permutation p-value (Laplace FGSM):", p_perm_lap_fgm)
print("Permutation p-value (Laplace PGD):", p_perm_lap_pgd)
print("Permutation p-value (Laplace CW):", p_perm_lap_cw)

#-------Effect Size--------

def cliffs_delta(clean, adv):
    n1, n2 = len(clean), len(adv)
    greater = sum(c > a for c in clean for a in adv)
    less    = sum(c < a for c in clean for a in adv)
    return (greater - less) / (n1 * n2)

print("\n======= Cliff's Δ (RBF)========= ")
print("Cliff's Δ (RBF FGSM):", cliffs_delta(rbf_clean, rbf_fgm))
print("Cliff's Δ (RBF PGD):", cliffs_delta(rbf_clean, rbf_pgd))
print("Cliff's Δ (RBF CW):", cliffs_delta(rbf_clean, rbf_cw))

print("\n======= Cliff's Δ (Laplace)========= ")
print("Cliff's Δ (Laplace FGSM):", cliffs_delta(lap_clean, lap_fgm))
print("Cliff's Δ (Laplace PGD):", cliffs_delta(lap_clean, lap_pgd))
print("Cliff's Δ (Laplace CW):", cliffs_delta(lap_clean, lap_cw))


##------- Detection Rate Computation--------##

###---- Fashion MNIST, RBF, Laplace, TPR, TNR -----###


from sklearn.metrics import roc_curve, auc

def compute_roc_from_scores(clean_scores, adv_scores):
    """
    clean_scores : deviation scores for clean samples
    adv_scores   : deviation scores for adversarial samples
    
    Returns:
        fpr, tpr, roc_auc
    """
    
    # Labels: 0 = clean (H0 accepted), 1 = adversarial (H0 rejected)
    y_true = np.concatenate([
        np.zeros(len(clean_scores)),
        np.ones(len(adv_scores))
    ])
    
    y_scores = np.concatenate([
        clean_scores,
        adv_scores
    ])
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    return fpr, tpr, roc_auc


""" Compute TPR, FPR for RBF """

fpr_fgsm_rbf, tpr_fgsm_rbf, auc_fgsm_rbf = compute_roc_from_scores(rbf_clean, rbf_fgm)
fpr_pgd_rbf,  tpr_pgd_rbf,  auc_pgd_rbf  = compute_roc_from_scores(rbf_clean, rbf_pgd)
fpr_cw_rbf,   tpr_cw_rbf,   auc_cw_rbf   = compute_roc_from_scores(rbf_clean, rbf_cw)


""" AUROC PLot for RBF Detection """

fig, ax = plt.subplots(figsize=(8,6))

ax.plot(fpr_fgsm_rbf, tpr_fgsm_rbf,
        label=f"FGSM (AUROC = {auc_fgsm_rbf:.3f})")
ax.plot(fpr_pgd_rbf, tpr_pgd_rbf,
        label=f"PGD (AUROC = {auc_fgsm_rbf:.3f})")
ax.plot(fpr_cw_rbf, tpr_cw_rbf,
        label=f"CW (AUROC = {auc_fgsm_rbf:.3f})")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("NPKD2_R Test Detection AUROC Scores")

ax.legend(loc="lower right")
ax.grid(True)

plt.tight_layout()
plt.show()


""" Compute TPR, FPR for Laplace function """

fpr_fgsm_lap, tpr_fgsm_lap, auc_fgsm_lap = compute_roc_from_scores(lap_clean, lap_fgm)
fpr_pgd_lap,  tpr_pgd_lap,  auc_pgd_lap  = compute_roc_from_scores(lap_clean, lap_pgd)
fpr_cw_lap,   tpr_cw_lap,   auc_cw_lap   = compute_roc_from_scores(lap_clean, lap_cw)



#-------- AUROC plots for Laplace Detection --------#

fig, ax = plt.subplots(figsize=(8,6))


ax.plot(fpr_fgsm_lap,tpr_fgsm_lap, label=f"FGSM (AUROC = {auc_fgsm_lap:.3f})" )
ax.plot(fpr_pgd_lap,  tpr_pgd_lap, label = f"PGD (AUROC = {auc_pgd_lap:.3f})")
ax.plot(fpr_cw_lap,   tpr_cw_lap, label = f"CW (AUROC = {auc_cw_lap:.3f})")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("NPKD2_L Test Detection AUROC Scores")
ax.legend(loc = "lower right")
ax.grid(True)
plt.tight_layout()
plt.show()



