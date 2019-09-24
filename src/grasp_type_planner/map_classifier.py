import numpy as np
from scipy.stats import multivariate_normal
from sklearn import mixture

class MapClassifier:
    def __init__(self, prob='gaus'):
        self.prob = prob
        self.label_prior = [None, None]
        if self.prob == 'gaus':
            self.gaus_mean = [None, None]
            self.gaus_cov = [None, None]
        elif self.prob == 'gmm':
            self.n_components = 2
            self.gmm = [None, None]
        #It's wierd the deprecated sklearn GMM and the new sklearn GMM has 
        #different performance!
        #The deprecated sklearn GMM performs better for now.
        self.deprecated_gmm = True
        if self.deprecated_gmm:
            import warnings
            warnings.filterwarnings("ignore", category=DeprecationWarning)
        #np.random.seed(1)

    def fit(self, X, y):
        if self.prob == 'gaus':
            self.__fit_gaus(X, y)
        elif self.prob == 'gmm':
            self.__fit_gmm(X, y)

    def predict_prob(self, X):
        if self.prob == 'gaus':
            y = self.__predict_prob_gaus(X)
        elif self.prob == 'gmm':
            y = self.__predict_prob_gmm(X)
        y = np.array(y).transpose()
        return y

    def predict(self, X):
        y = predict_prob(X)
        return (y[:, 1] > 0.5).astype(int)

    def __fit_label_priors(self, y):
        n_samples = y.shape[0]
        self.label_prior[1] = float(np.sum(y)) / n_samples 
        self.label_prior[0] = 1 - self.label_prior[1]

    def __fit_single_gaus(self, X):
        '''
        MLE to fit a Gaussian distribution.
        '''
        n_samples, n_dim = X.shape
        gaus_mean = np.mean(X, axis=0)
        X_bar = X - gaus_mean 
        gaus_cov = np.matmul(X_bar.transpose(), X_bar) / (n_samples - 1)
        return gaus_mean, gaus_cov

    def __fit_gaus(self, X, y):
        '''
        Fit Gaussian for positive and negative grasps seperately.
        '''
        #Notice: y needs to be a numpy array
        for i in xrange(2):
            X_seperate = X[y==i] 
            self.gaus_mean[i], self.gaus_cov[i] = self.__fit_single_gaus(X_seperate)
        self.__fit_label_priors(y)
        #print self.gaus_mean[0]
        #print self.gaus_cov[0]
        #print self.gaus_mean[1]
        #print self.gaus_cov[1]

    def __predict_prob_gaus(self, X):
        '''
        Map inference for grasp success label, like Eigenobject.
        '''
        #Label posterior distribution.
        label_posterior = [None, None]
        #Change to log probability to avoid numerical issues?
        for i in xrange(2):
            gaus = multivariate_normal(mean=self.gaus_mean[i], cov=self.gaus_cov[i], allow_singular=True)
            likelihood = gaus.pdf(X)
            if X.shape[0] == 1:
                likelihood = np.array([likelihood])
            label_posterior[i] = self.label_prior[i] * likelihood
        #print 'Posterior without normalization:', label_posterior
        #There will be numerical issues when the total posterior is zero.
        #Treat as failure for such cases now. There should be a better way to 
        #handle this. 
        label_posterior /= np.sum(label_posterior, axis=0)
        label_posterior = np.nan_to_num(label_posterior)
        #print 'Posterior with normalization:', label_posterior
        return label_posterior

    def __fit_gmm(self, X, y):
        '''
        Fit GMM for positive and negative grasps seperately.
        '''
        init_method = 'kmeans'
        #init_method = 'random'
        #Notice: y needs to be a numpy array
        for i in xrange(2):
            X_seperate = X[y==i] 
            #Notice: looks like gmm learning is very sensitive to the random_state!
            #There is still randomness even if random_state is fixed. Why?
            if not self.deprecated_gmm:
                g = mixture.GaussianMixture(n_components=self.n_components, covariance_type='full', 
                        random_state=0, init_params=init_method, n_init=5)
            else:
                #g = mixture.GMM(n_components=self.n_components, covariance_type='full', 
                #        random_state=0)
                g = mixture.GMM(n_components=self.n_components, covariance_type='full', 
                        random_state=100000, n_init=5)

            g.fit(X_seperate)
            self.gmm[i] = g 
        self.__fit_label_priors(y)
        #if not self.deprecated_gmm:
        #    print self.gmm[0].get_params()
        #    print self.gmm[0].weights_
        #    print self.gmm[0].means_
        #    print self.gmm[0].covariances_
        #    print self.gmm[1].get_params()
        #    print self.gmm[1].weights_
        #    print self.gmm[1].means_
        #    print self.gmm[1].covariances_
        #else:
        #    print self.gmm[0].get_params()
        #    print self.gmm[0].weights_
        #    print self.gmm[0].means_
        #    print self.gmm[0].covars_
        #    print self.gmm[1].get_params()
        #    print self.gmm[1].weights_
        #    print self.gmm[1].means_
        #    print self.gmm[1].covars_

    def __predict_prob_gmm(self, X):
        '''
        Map inference for grasp success label, like Eigenobject.
        '''
        #Label posterior distribution.
        label_posterior = [None, None]
        for i in xrange(2):
            #likelihood = self.gmm[i].predict_proba(X)
            if not self.deprecated_gmm:
                log_prob = self.gmm[i].score_samples(X)
            else:
                 log_prob, _ = self.gmm[i].score_samples(X)
            #print log_prob
            likelihood = np.exp(log_prob)
            label_posterior[i] = self.label_prior[i] * likelihood
        #print 'Posterior without normalization:', label_posterior
        label_posterior /= np.sum(label_posterior, axis=0)
        label_posterior = np.nan_to_num(label_posterior)
        #print 'Posterior with normalization:', label_posterior
        return label_posterior


