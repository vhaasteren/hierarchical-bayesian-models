# prior_wrapper.py
"""Classes and functions that can wrap an Enterprise pta object for HBMs

Simple usage instructions
-------------------------
Build your Enterprise PTA object as you normally would. Make sure you know what
the parameter names are for the signals that you want to place a hierarchical
prior on. Those parameter names will be matched with a regular expression.
For instance, for powerlaw red noise, often the parameters would be named:
'J0030+0451_red_noise_log10_A'
'J0030+0451_red_noise_gamma'
'J1713+0747_red_noise_log10_A',
...

With the above naming convention, you could create an HBM like this:

import prior_wrapper as pw

wrapper = pw.EnterpriseWrapper(
    pta=pta,
    hyper_regexps = {
        'red_noise': {
            'log10_amp': '_red_noise_log10_A$',
            'gamma': '_red_noise_gamma$',
            'prior': pw.BoundedMvNormalPlHierarchicalPrior,
        }
    }
)

x0 = wrapper.sample()
cov = np.diag(0.01 * np.ones_like(x0))

sampler = ptmcmc(len(x0), wrapper.log_likelihood, wrapper.log_prior, cov, outDir='./chains', resume=False)

# parameter names are in:
# wrapper.param_names

# Add the prior draws
for draw_function in wrapper.get_draw_from_prior_functions():
    sampler.addProposalToCycle(draw_function, 5)


# If you also want to model DMGP with a Hierarchical Prior, you could do:

wrapper = pw.EnterpriseWrapper(
    pta=pta,
    hyper_regexps = {
        'red_noise': {
            'log10_amp': '_red_noise_log10_A$',
            'gamma': '_red_noise_gamma$',
            'prior': pw.BoundedMvNormalPlHierarchicalPrior,
        },
        'dmgp': {
            'log10_amp': '_dm_gp_log10_A$',
            'gamma': '_dm_gp_gamma$',
            'prior': pw.BoundedMvNormalPlHierarchicalPrior,
        },
    }
)

# This last one is largely untested. Since the prior ranges for the
# hyperparameters cannot be set dynamically at the moment, we may need to update
# the code a bit to make it work for more general priors

"""

import numpy as np
import scipy.stats as sstats
import scipy.linalg as sl
import re
from enterprise.signals.parameter import function
import enterprise.constants as const

class kumaraswamy_distribution(sstats.rv_continuous):
    """Kumaraswamy distribution like for scipy"""
    def _pdf(self, x, a, b):
        return np.where((x >= 0) & (x <= 1), a * b * x**(a-1) * (1 - x**a)**(b-1), 0)

    def _cdf(self, x, a, b):
        return np.where((x >= 0) & (x <= 1), 1 - (1 - x**a)**b, 0)

    def _ppf(self, q, a, b):
        return (1 - (1 - q)**(1.0/b))**(1.0/a)

def get_sigmoid_transformed_distribution(distribution, lower=0, upper=1, name='sigmoid-transformed-distribution'):
    """Class factory for a sigmoid-transformed scipy distribution"""

    def forward(x):
        return np.log( (x-lower) / (upper-x))

    def backward(p):
        num = (upper-lower) * np.exp(p)
        den = 1 + np.exp(p)
        return lower + num/den

    def dpdx(x):
        num = (x-lower) + (upper-x)
        den = (x-lower) * (upper-x)
        return num / den

    def log_dpdx(x):
        num = (x-lower) + (upper-x)
        den = (x-lower) * (upper-x)
        return np.log(num) - np.log(den)

    def dxdp(p):
        num = 1 + np.exp(p)
        return (upper-lower) * (1/num - 1/(num**2))


    class sigmoid_transformed_distribution(sstats.rv_continuous):

        def _pdf(self, x, **kwargs):
            dist = distribution(**kwargs)
            p = forward(x)
            return np.exp(dist.logpdf(p) + log_dpdx(x))

        def _logpdf(self, x, **kwargs):
            dist = distribution(**kwargs)
            p = forward(x)
            return dist.logpdf(p) + log_dpdx(x)

        def _pdf(self, x, **kwargs):
            return np.exp(self._logpdf(x, **kwargs))

        def _logcdf(self, x, **kwargs):
            dist = distribution(**kwargs)
            p = forward(x)
            return dist.logcdf(p)

        def _cdf(self, x, **kwargs):
            return np.exp(self._logcdf(x, **kwargs))

        def _ppf(self, u, **kwargs):
            dist = distribution(**kwargs)
            p = dist.ppf(u)
            return backward(p)

    return sigmoid_transformed_distribution(name=name)

# Create an instance of the Kumaraswamy distribution
kumaraswamy = kumaraswamy_distribution(name='kumaraswamy')


class IntervalTransform(object):
    """Stand-alone interval Transform class"""
    def __init__(self, lower=0, upper=1):
        self._lower = lower
        self._upper = upper

    def forward(self, x):
        return np.log( (x-self._lower) / (self._upper-x))

    def backward(self, p):
        num = (self._upper-self._lower) * np.exp(p)
        den = 1 + np.exp(p)
        return self._lower + num/den

    def dpdx(self, x):
        num = (x-self._lower) + (self._upper-x)
        den = (x-self._lower) * (self._upper-x)
        return num / den

    def log_dpdx(self, x):
        num = (x-self._lower) + (self._upper-x)
        den = (x-self._lower) * (self._upper-x)
        return np.log(num) - np.log(den)

    def dxdp(self, p):
        num = 1 + np.exp(p)
        return (self._upper-self._lower) * (1/num - 1/(num**2))

    def log_dxdp(self, p):
        num = 1 + np.exp(p)
        return np.log(self._upper-self._lower) + np.log(1/num - 1/(num**2))

def log_sum_exp(log_prior1, log_prior2):
    """Take the log of two exponential sums, stable numerically"""
    max_log_prior = np.maximum(log_prior1, log_prior2)
    log_prior_diff = np.abs(log_prior1 - log_prior2)

    # Compute the sum of exponentials with numerical stability
    log_sum_exp_result = max_log_prior + np.log(1 + np.exp(-log_prior_diff))

    return log_sum_exp_result

def log_weighted_sum_exp(log_prior1, log_prior2, f):
    max_log_prior = np.maximum(log_prior1, log_prior2)
    #log_prior_diff = np.abs(log_prior1 - log_prior2)
    
    # Compute the sum of exponentials with numerical stability
    log_sum_exp_result = max_log_prior + np.log(f * np.exp(log_prior1 - max_log_prior) + (1 - f) * np.exp(log_prior2 - max_log_prior))
    
    return log_sum_exp_result

def ptapar_offsets(pta):
    """Some parameters in Enterprise are an array. Get the offsets"""
    sizes = [1 if p.size is None else p.size for p in pta.params]
    return np.hstack([[0], np.cumsum(sizes)[:-1]])

def ptapar_mapping(pta):
    """Create a mapping between arrays, and enterprise parameters"""
    sizes = [1 if p.size is None else p.size for p in pta.params]
    offsets = np.hstack([[0], np.cumsum(sizes)])

    ptapar_to_array = [np.arange(offsets[ii],offsets[ii+1]) for ii in range(len(pta.params))]
    array_to_ptapar = np.array([ii for (ii, inds) in enumerate(ptapar_to_array) for _ in inds])

    return ptapar_to_array, array_to_ptapar

@function
def powerlaw_flat_tail(f, log10_A=-16, gamma=5, log10_kappa=-7, components=2):
    df = np.diff(np.concatenate((np.array([0]), f[::components])))
    return (
        (10**log10_A) ** 2 / 12.0 / np.pi**2 * const.fyr ** (gamma - 3) * f ** (-gamma) * np.repeat(df, components) + 10 ** (2*log10_kappa)
    )

class BoundedMvNormalPlHierarchicalPrior(object):
    """Class to represent a Bounded MvNormal hierarchical prior for Enterprise Powerlaw Signals"""

    def __init__(self, pta, log_amplitude_regexp, gamma_regexp, ind_offset=0, gamma_lower=0, gamma_upper=7, name='MvNormal'):
        """This is a Hierarchical prior component for use with the EnterpriseWrapper
        
        This class represents a single MvGaussian hyper-prior on a set of parameters. It was greated for power-law parameters, for which one of the two parameters has a bounded interval: gamma has [0,7] as its domain

        The parameters are selected through regular expressions. For powerlaw noise, you would do:
        log_amplitude_regexp = r"_red_noise_log10_A$"
        gamma_regexp = r"_red_noise_gamma$"

        For DM variations you would do:
        log_amplitude_regexp = r"_dm_gp_log10_A$"
        gamma_regexp = r"_dm_gp_gamma$"
        """

        self._pta = pta
        self._la_pattern = re.compile(log_amplitude_regexp)
        self._g_pattern = re.compile(gamma_regexp)

        # Select the relevant parameters of Enterprise
        la_param_names = list(filter(self._la_pattern.search, pta.param_names))
        g_param_names = list(filter(self._g_pattern.search, pta.param_names))

        # Parameter masks
        self._la_msk = np.array([pn in la_param_names for pn in pta.param_names], dtype=bool)
        self._g_msk = np.array([pn in g_param_names for pn in pta.param_names], dtype=bool)

        self._la_inds = np.where(self._la_msk)[0]
        self._g_inds = np.where(self._g_msk)[0]
        self._ind_offset = ind_offset
        self._ind_hyper = np.arange(ind_offset, ind_offset+self.hyper_ndim())
        self._ind_level1 = np.sort(np.hstack([self._la_inds, self._g_inds], dtype=int))

        if len(self._la_inds) != len(self._g_inds):
            raise ValueError("Unequal number of amplitude / gamma parameters")

        self._gamma_transform = IntervalTransform(lower=gamma_lower, upper=gamma_upper)
        self._gamma_lower = gamma_lower
        self._gamma_upper = gamma_upper
        self._npsrs = len(self._la_inds)
        self._name = name

        self.set_hyperpriors()

    def hyper_ndim(self):
        """The number of hyperparameters of this prior class"""
        return 5

    def set_hyperpriors(self):
        """Set the hyper parameter priors"""
        # TODO: allow for more flexible setting of hyperparameter ranges
        self._mu_amp = sstats.uniform(loc=-20, scale=10)
        self._mu_gamma = sstats.uniform(loc=-4, scale=8)
        self._L_amp = sstats.uniform(loc=0.03, scale=3.47)
        self._L_gamma = sstats.uniform(loc=0.03, scale=3.47)
        self._L_12 = sstats.uniform(loc=-1.5, scale=3)

        self._hyper_dists = [
            self._mu_amp,
            self._mu_gamma,
            self._L_amp,
            self._L_gamma,
            self._L_12,
        ]

        self.hyperparameter_names = [
            f"{self._name}_mu_amp",
            f"{self._name}_mu_gamma",
            f"{self._name}_L_A",
            f"{self._name}_L_gamma",
            f"{self._name}_L_12",
        ]

    def get_hyper_pars(self, x):
        """Get only the hyperparameters"""
        return np.array(x)[self._ind_hyper]

    def get_mu_L(self, x):
        """Get the mu and L of the prior"""
        (mu1, mu2, L1, L2, L12) = self.get_hyper_pars(x)

        L = np.array([[L1, 0],[L12, L2]])   # The Cholesky decomposition
        mu = np.array([mu1, mu2])
        return mu, L

    def forward(self, x):
        """Transform the gamma parameters to their transformed state"""
        p = np.copy(x)
        p[self._g_inds] = self._gamma_transform.forward(p[self._g_inds])
        return p

    def backward(self, p):
        """Transform the gamma parameters back to their original state"""
        x = np.copy(p)
        x[self._g_inds] = self._gamma_transform.backward(x[self._g_inds])
        return x

    def log_dpdx(self, x):
        """The Jacobian of the gamma transform"""
        return np.sum([self._gamma_transform.log_dpdx(x[gi]) for gi in self._g_inds])

    def log_hyperprior(self, x):
        """The hyperprior log-prior"""
        hyper_pars = self.get_hyper_pars(x)

        return np.sum([dd.logpdf(pp) for (dd, pp) in zip(self._hyper_dists, hyper_pars)])

    def sample(self):
        """Draw a random sample from this prior"""
        x0 = np.zeros(self._ind_offset + self.hyper_ndim())

        x_hyper = np.array([d.ppf(np.random.rand()) for d in self._hyper_dists])
        x0[self._ind_hyper] = x_hyper

        mu, L = self.get_mu_L(x0)
        uag = np.random.randn(2, self._npsrs)
        xag = mu[:,None] + np.dot(L, uag)

        x0[self._la_inds] = xag[0,:]
        x0[self._g_inds] = np.clip(xag[1,:], self._gamma_lower+0.001, self._gamma_upper-0.001)

        return x0[np.concatenate([self._ind_level1, self._ind_hyper])]

    def get_level1_parameter_mask(self):
        return np.logical_or(self._la_msk, self._g_msk)

    def get_parameter_inds(self):
        return np.sort(np.hstack([self._ind_level1, self._ind_hyper]))
    
    def get_hyperparameter_names(self):
        return self.hyperparameter_names

    def log_prior(self, x):
        """The full prior, including all HBM levels, for this component"""
        x_gammas = x[self._g_inds]
        if np.any(x_gammas <= self._gamma_lower) or np.any(x_gammas >= self._gamma_upper):
            return -np.inf

        p = self.forward(x)
        mu, L = self.get_mu_L(x)

        amps = p[self._la_inds]
        gammas = p[self._g_inds]
        pag = np.vstack([amps, gammas])
        try:
            uag = sl.solve_triangular(L, pag - mu[:,None], trans=0, lower=True)
        except sl.LinAlgError as e:
            return -np.inf

        quad = -0.5 * np.sum(uag**2, axis=0)
        norm = - np.sum(np.log(np.diag(L))) - np.log(2*np.pi)
        log_prior = np.sum(quad + norm)
        log_jacobian = self.log_dpdx(x)
        log_hyperprior = self.log_hyperprior(x)

        return log_prior + log_jacobian + log_hyperprior

    def get_draw_from_priors(self, full_log_prior):
        """Create prior draw functions for PTMCMC"""

        def draw_from_mvn_prior_hyper(x, iter, beta):
            """Draw a new hyperparameter"""
            q = x.copy()

            # Select random parameter to jump in & propose
            ind = np.random.randint(0, self.hyper_ndim())
            ind_off = ind + self._ind_offset
            prior_dist = self._hyper_dists[int(ind)]
            q[ind_off] = prior_dist.rvs()

            # Use only the hyper-prior here, as that's what we draw from
            # The full prior is conditional on these parameters, so it
            # would otherwise change a lot
            x_logp = prior_dist.logpdf(x[ind_off])
            q_logp = prior_dist.logpdf(q[ind_off])

            lqxy = x_logp - q_logp

            return q, float(lqxy)

        def draw_from_mvn_prior_low(x, iter, beta):
            """Draw a new low-level parameter from the conditional MvGaussian"""
            q = x.copy()

            # Go to transformed coordinates
            qp = self.forward(q)

            # Whiten the parameters
            mu, L = self.get_mu_L(x)
            amps = qp[self._la_inds]
            gammas = qp[self._g_inds]
            pag = np.vstack([amps, gammas])
            try:
                uag = sl.solve_triangular(L, pag - mu[:,None], trans=0, lower=True)
            except sl.LinAlgError as e:
                return x, 0

            # Draw a random element from uag to update
            n_total = np.prod(uag.shape)
            random_index = np.random.choice(n_total)
            indices = np.unravel_index(random_index, uag.shape)

            # Draw a new value for this parameter
            uag[indices] = np.random.randn()
            pag = mu[:,None] + np.dot(L, uag)

            # Transform back to original coordinates
            qp[self._la_inds] = pag[0,:]
            qp[self._g_inds] = pag[1,:]
            q = self.backward(qp)

            x_logp = full_log_prior(x)
            q_logp = full_log_prior(q)

            lqxy = x_logp - q_logp

            return q, float(lqxy)

        return draw_from_mvn_prior_hyper, draw_from_mvn_prior_low
        #return (draw_from_mvn_prior_low,)

class BoundedTwoComponentMvNormalPlHierarchicalPrior(BoundedMvNormalPlHierarchicalPrior):
    """Same as BoundedMvNormalPlHierarchicalPrior, but with two populations"""

    def __init__(self, pta, log_amplitude_regexp, gamma_regexp, ind_offset=0, gamma_lower=0, gamma_upper=7, name='TwoComponentMvNormal'):
        super().__init__(
                        pta,
                        log_amplitude_regexp,
                        gamma_regexp,
                        ind_offset,
                        gamma_lower=gamma_lower,
                        gamma_upper=gamma_upper,
                        name=name
            )
        
    def hyper_ndim(self):
        return 11

    def set_hyperpriors(self):
        super().set_hyperpriors()

        # Perhaps this should be the Beta(0.5, 0.5) distribution?
        self._fraction = sstats.uniform(loc=0.1, scale=0.8)
        self._hyper_dists = self._hyper_dists + self._hyper_dists + [self._fraction]

        self.hyperparameter_names = [
            f"{self._name}_mu1_amp",
            f"{self._name}_mu1_gamma",
            f"{self._name}_L1_A",
            f"{self._name}_L1_gamma",
            f"{self._name}_L1_12",
            f"{self._name}_mu2_amp",
            f"{self._name}_mu2_gamma",
            f"{self._name}_L2_A",
            f"{self._name}_L2_gamma",
            f"{self._name}_L2_12",
            f"{self._name}_CF",
        ]

    def get_all_mu_L(self, x):
        """Get the mu and L of the prior"""
        (mu1_1, mu1_2, L1_1, L1_2, L1_12, mu2_1, mu2_2, L2_1, L2_2, L2_12, CF) = self.get_hyper_pars(x)

        L1 = np.array([[L1_1, 0],[L1_12, L1_2]])   # The Cholesky decomposition
        mu1 = np.array([mu1_1, mu1_2])

        L2 = np.array([[L2_1, 0],[L2_12, L2_2]])   # The Cholesky decomposition
        mu2 = np.array([mu2_1, mu2_2])
        return mu1, L1, mu2, L2, CF

    def get_mu_L(self, x):
        """Get the mu and L for a random component (with prob CF)

        This is done, so that we can keep the following functions from superclass
        - get_draw_from_priors
        - sample
        """

        mu1, L1, mu2, L2, CF = self.get_all_mu_L(x)

        if np.random.rand() <= CF:
            return mu1, L1
        
        else:
            return mu2, L2

    def log_prior(self, x):
        """The full prior, including all HBM levels, for this component"""

        x_gammas = x[self._g_inds]
        if np.any(x_gammas <= self._gamma_lower) or np.any(x_gammas >= self._gamma_upper):
            return -np.inf

        # Hyper parameter quantities
        p = self.forward(x)
        mu1, L1, mu2, L2, CF = self.get_all_mu_L(x)

        # Demand that mu1 > mu2 (gamma), so that there is no mode confusion?
        if mu1[1] > mu2[1]:
            return -np.inf

        # Amplitudes & Gammas
        amps = p[self._la_inds]
        gammas = p[self._g_inds]
        pag = np.vstack([amps, gammas])

        # Mode 1 & 2 Gaussian components
        try:
            uag1 = sl.solve_triangular(L1, pag - mu1[:,None], trans=0, lower=True)
            quad1 = -0.5 * np.sum(uag1**2, axis=0)
            norm1 = - np.sum(np.log(np.diag(L1))) - np.log(2*np.pi)
            uag2 = sl.solve_triangular(L2, pag - mu2[:,None], trans=0, lower=True)
            quad2 = -0.5 * np.sum(uag2**2, axis=0)
            norm2 = - np.sum(np.log(np.diag(L2))) - np.log(2*np.pi)
            log_prior1 = np.sum(quad1 + norm1)
            log_prior2 = np.sum(quad2 + norm2)
        except sl.LinAlgError as e:
            return -np.inf

        log_prior = log_weighted_sum_exp(log_prior1, log_prior2, CF)
        log_jacobian = self.log_dpdx(x)
        log_hyperprior = self.log_hyperprior(x)

        return log_prior + log_jacobian + log_hyperprior


class EnterpriseWrapper(object):
    """Class to wrap an Enterprise pta object to allow for Hierarchical Priors"""

    def __init__(self, pta, hyper_regexps={}):
        """Initialize the Enterprise Wrapper

        param hyper_regexps: dict of dictionaries
                             {'rn_noise': {'log10_amp': 'regexp',
                                           'gamma': 'regexp',
                                           'prior': BoundedMvNormalPlHierarchicalPrior}}

        """
        self._pta = pta
        self._ndim_level1 = len(pta.param_names)
        self._ndim_level2 = 0
        self.hyper_priors = []

        for noise_component_name, noise_component in hyper_regexps.items():
            prior_class = noise_component.get('prior', BoundedMvNormalPlHierarchicalPrior)
            prior = prior_class(pta,
                                noise_component['log10_amp'],
                                noise_component['gamma'],
                                ind_offset=self._ndim_level1+self._ndim_level2,
                                gamma_lower=0,
                                gamma_upper=7,
                                name=noise_component_name
                )

            self._ndim_level2 += prior.hyper_ndim()

            self.hyper_priors.append(prior)

        self._ndim = self._ndim_level1 + self._ndim_level2
        self._ptapar_to_array, self._array_to_ptapar = ptapar_mapping(self._pta)

        # Initialize all the Enterprise prior distributions for efficiency
        self._nohbm_indices = self.get_nohbm_indices()
        nohbm_parameter_indices = list(set(self._array_to_ptapar[self._nohbm_indices]))
        self._nohbm_parameters = [self._pta.params[pp] for pp in nohbm_parameter_indices]

    @property
    def param_names(self):
        """All parameter names of whole HBM"""
        param_names_orig = self._pta.param_names

        param_names_hyper = [hp for prior in self.hyper_priors for hp in prior.hyperparameter_names]

        return param_names_orig + param_names_hyper

    def hbm_level1_indices(self):
        """All the level1 (not flat, not level2) indices"""
        level1_indices = []

        for prior in self.hyper_priors:
            level1_indices = np.concatenate([level1_indices, np.where(prior.get_level1_parameter_mask())[0]])

        return np.sort(np.hstack(level1_indices))

    def hbm_level2_indices(self):
        """All HBM level2 indices"""
        level2_indices = []

        for prior in self.hyper_priors:
            level2_indices = np.concatenate([level2_indices, prior._ind_hyper])

        return np.sort(np.hstack(level2_indices))

    def get_nohbm_indices(self):
        """All indices that are not multi-level"""
        all_inds = set(np.arange(self._ndim))

        return np.array(list(all_inds - set(self.hbm_level1_indices()) - set(self.hbm_level2_indices())))

    def sample_orig(self):
        """Sample from the original Enterprise prior, not the Hierarchical one"""
        x0_orig = np.hstack([p.sample() for p in self._pta.params])
        return x0_orig

    def sample(self):
        """Sample randomly from the HBM prior"""
        x = np.zeros(self._ndim)
        x_orig = self.sample_orig()
        x[:len(x_orig)] = x_orig        # Some will be overwritten

        while True:
            for prior in self.hyper_priors:
                x_prior = prior.sample()
                x[prior.get_parameter_inds()] = x_prior

            if np.isfinite(self.log_prior(x)):
                return x

        return x

    def get_low_level_pars(self, x):
        """Return only the low-level parameters. Includes *flat* parameters"""
        return x[:self._ndim_level1]

    def log_prior(self, x):
        """Full hierarchical log-prior"""
        params = self._pta.map_params(self.get_low_level_pars(x))
        logp = np.sum([p.get_logpdf(params=params) for p in self._nohbm_parameters])

        for prior in self.hyper_priors:
            logp += prior.log_prior(x)

        return logp

    def log_likelihood(self, x):
        """Log-likelihood as defined by Enterprise"""
        try:
            logp = self._pta.get_lnlikelihood(self.get_low_level_pars(x))
        except (ValueError, OverflowError):
            logp = -np.inf

        return logp

    def get_draw_from_prior_functions(self):
        """Create a list of prior draw functions for PTMCMC"""

        def draw_from_prior_flat(x, iter, beta):
            """Draw from flat (non-multi-level) parameters"""
            q = x.copy()

            ind = np.random.choice(self.get_nohbm_indices())
            q[ind] = np.atleast_1d(self._pta.params[self._array_to_ptapar[ind]].sample())[0]

            x_logp = self.log_prior(x)
            q_logp = self.log_prior(q)

            lqxy = x_logp - q_logp

            return q, float(lqxy)

        prior_functions = [draw_from_prior_flat]

        for prior in self.hyper_priors:
            for draw_function in prior.get_draw_from_priors(self.log_prior):
                prior_functions.append(draw_function)

        return prior_functions
