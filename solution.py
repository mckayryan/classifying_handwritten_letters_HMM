########################## Authors ############################
###############     Mattia Gentil z5178386      ###############
###############     Danial Khosravi z5054176    ###############
###############     Ryan McKay z5060961         ###############
###############################################################

########################## Libraries ##########################

import scipy.io as sio
from scipy.misc import logsumexp
import numpy as np
import hmmlearn
from scipy.signal import butter, lfilter, freqz
from hmmlearn.hmm import GaussianHMM
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import warnings


# due to https://github.com/hmmlearn/hmmlearn/issues/158 and
# https://github.com/hmmlearn/hmmlearn/issues/175
# the fitting process is going to give a LOT of warnings
warnings.filterwarnings('ignore')


seed = 1234
rng = np.random.RandomState(seed)

########################## Load the data ##########################

trainData = sio.loadmat('./trajectories_train.mat')
testData = sio.loadmat('./trajectories_xtest.mat')


######################## Data preprocessing ########################

xtrain = trainData['xtrain'].reshape((-1, ))
ytrain = trainData['ytrain'].reshape((-1, ))
xtest = testData['xtest'].reshape((-1, ))

xtrain = np.asarray([seq.T for seq in xtrain])
xtest = np.asarray([seq.T for seq in xtest])

# finding the min and max of each dimension accross the whole training data
train_max = np.max(np.vstack(xtrain), axis=0)
train_min = np.min(np.vstack(xtrain), axis=0)

def rescale(seq):
    return (seq - train_min) / (train_max - train_min)

xtrain = np.asarray([rescale(seq) for seq in xtrain])
xtest = np.asarray([rescale(seq) for seq in xtest])

def butter_lowpass(cutoff, fs, order=5):

    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):

    b, a = butter_lowpass(cutoff, fs, order=order)
    filtered = np.empty_like(data)

    for i in range(filtered.shape[0]):
        ll = np.zeros(data[i].shape)

        for j in range(3):
            ll[:, j] = lfilter(b, a, data[i][:, j])

        filtered[i] = ll

    return filtered

# Filter hyper-parameters
order = 4
fs = 200.0  # sample rate, Hz
cutoff = 3  # desired cutoff frequency of the filter, Hz

# Applying the low pass filter to the data
xtrain = butter_lowpass_filter(xtrain, cutoff, fs, order)
xtest = butter_lowpass_filter(xtest, cutoff, fs, order)

# Encoding the labels
label_enc = LabelEncoder().fit(ytrain)


######################## HMM Generative Classifier ########################

# Utility functions to calculate the log_likelihood of sequences

def log_likelihood(hmm, sequence):

    logprob_frame = hmm._compute_log_likelihood(sequence)
    logprob_sequence, _ = hmm._do_forward_pass(logprob_frame)

    return logprob_sequence

def log_likelihoods(hmm, sequences):

    ll = lambda seq: log_likelihood(hmm, seq)

    return np.fromiter(map(ll, sequences), dtype='float64')

def log_likelihoods_cond(cond_hmms, sequences):

    ll = lambda hmm: log_likelihoods(hmm, sequences)

    return np.vstack(map(ll, cond_hmms))



class GenerativeClassifierHMM(BaseEstimator, ClassifierMixin):
    '''
    GenerativeClassifierHMM trains a HMM for each of the class-conditional subsets of 
    the training data and nominate the class corresponding to the class conditional HMM
    with the highest log-likelihood for prediction.
    It can also use a different number of states for each of the class conditionals.
    '''
    def __init__(self, hmm=GaussianHMM()):
        self.hmm = hmm
        self.class_cond_hmms_ = []

    def fit(self, sequences, labels, class_cond_states):

        class_counts = np.bincount(labels).astype(np.float)
        self.logprior = np.log(class_counts / np.sum(class_counts))

        for c in range(np.max(labels)+1):
            sequences_c = sequences[labels == c]
            X_c = np.vstack(sequences_c)
            lengths_c = list(map(len, sequences_c))
            class_cond_hmm = clone(self.hmm, safe=True)
            n_states_k = class_cond_states[c]
            pi0 = np.eye(1, n_states_k)[0]
            trans0 = np.diag(np.ones(n_states_k)) + np.diag(np.ones(n_states_k-1), 1)
            trans0 /= trans0.sum(axis=1).reshape(-1, 1)
            class_cond_hmm.n_components = n_states_k
            class_cond_hmm.startprob_ = pi0
            class_cond_hmm.transmat_ = trans0
            class_cond_hmm.fit(X_c, lengths=lengths_c)
            self.class_cond_hmms_.append(class_cond_hmm)

        return self

    def predict(self, sequences):

        log_likelihood_ = log_likelihoods_cond(self.class_cond_hmms_, sequences)
        log_post_unnorm = log_likelihood_ + self.logprior.reshape(-1, 1)

        return np.argmax(log_post_unnorm, axis=0)

    def predict_proba(self, sequences):

        log_likelihood_ = log_likelihoods_cond(self.class_cond_hmms_, sequences)
        log_post_unnorm = log_likelihood_ + self.logprior.reshape(-1, 1)
        prob_post_norm = np.empty_like(log_post_unnorm)

        for i in range(log_post_unnorm.shape[1]):
            prob_post_norm[:,i] = log_post_unnorm[:,i] - logsumexp(log_post_unnorm[:,i].astype(np.float64))
                
        return prob_post_norm

    def generateSample(self, mClass, length):

        sel_hmm = self.class_cond_hmms_[mClass]
        x, _ = sel_hmm.sample(length)

        return x


########################### Prediction ############################


hmm = GaussianHMM(n_iter=10, random_state=seed)
hmm_classifier = GenerativeClassifierHMM(hmm)
hmm_classifier.fit(xtrain, label_enc.transform(ytrain), np.tile(10, 20))

test_pred_probs = hmm_classifier.predict_proba(xtest)
predictions = pd.DataFrame(test_pred_probs.T)
predictions.to_csv('./predictions.txt', header=False, index=False)

