from collections import defaultdict
from lzma import open as lzma_open

from iminuit import Minuit
from pickle import dump as pickle_dump
from pickle import load as pickle_load
from torch import tensor, int32, float32, int64, float64, set_num_threads, set_num_interop_threads
from ultranest import ReactiveNestedSampler
from uproot import open as open_root
import numpy as np
from scipy.linalg import solve_triangular

from neuromct.configs import data_configs
from neuromct.fit import SamplerMH, LogLikelihood, LogLikelihoodRatio, NegativeLogLikelihood
from neuromct.models.ml import setup

import os
os.environ['MKL_NUM_THREADS'] = '1'
set_num_threads(1)
set_num_interop_threads(1)

sources_all = ('Cs137', 'K40', 'Co60', 'AmBe', 'AmC',)
source_to_number = dict(
        [(source_name, source_number) for source_number, source_name in enumerate(sources_all)]
        )

def get_conditions(n, path_to_model=None):
    # params space
    kB = np.arange(6.45, 24, 1.8)
    fC = np.arange(0.025, 1, 0.1)
    LY = np.arange(8100, 12000, 400)

    kB_list = []
    fC_list = []
    LY_list = []

    for i in range(10):
        for j in range(10):
            for k in range(10):
                kB_list.append(kB[i])
                fC_list.append(fC[j])
                LY_list.append(LY[k])

    kB_list.append(12.05)
    fC_list.append(0.517)
    LY_list.append(9846)

    kB_list = np.array(kB_list).reshape(-1, 1)
    fC_list = np.array(fC_list).reshape(-1, 1)
    LY_list = np.array(LY_list).reshape(-1, 1)
    conditions = np.concatenate([kB_list, fC_list, LY_list], axis=1)

    path_to_model = path_to_model if path_to_model is not None else data_configs['path_to_models']
    scaler = pickle_load(open(f"{path_to_model}/minmax_scaler.pkl", "rb"))

    conditions_default_reshaped = np.array(conditions[n]).reshape(1, -1)
    conditions_default_scaled = scaler.transform(conditions_default_reshaped)
    return conditions_default_scaled

def sort_sources_by_number(sources):
    return sorted(sources, key=lambda name: source_to_number[name])

def read_data(data_path, sources):
    edges = data_configs['kNPE_bins_edges']
    data_dict = dict()
    for source_name in sources:
        with open_root(f'{data_path}/{source_name}.root') as rootfile:
            energy = 1.07 * rootfile['TRec']['m_NPE'].array() / 1000.
            data, _ = np.histogram(energy, bins=edges)
            data_dict[source_name] = data
    return data_dict

class Model:
    def __init__(self, source, integral,
            bin_width=0.02, cholesky_cov=None, model_type='tede', device='cpu', path_to_models=None):
        self._source_n = source_to_number[source]
        self._integral = integral
        self._bin_width = bin_width
        self._cholesky_cov = cholesky_cov
        self._model = setup(model_type, device, path_to_models)

    def __call__(self, pars):
        kb, fc, ly, n = pars
        if self._cholesky_cov is not None:
            kb, fc, ly = self._cholesky_cov @ [kb, fc, ly]
        nl_pars = tensor([kb, fc, ly], dtype=float64).unsqueeze(0)
        out = self._model(
                nl_pars, tensor([[self._source_n]], dtype=int64)
                ).detach().numpy()[0] * self._integral * self._bin_width
        return n * out

def cost_funs_sum_wrapper(cost_funs):
    def cost_funs_sum(pars):
        res = 0
        kb, fc, ly = pars[:3]
        norm_consts = pars[3:]
        for (n, cost_fn) in zip(norm_consts, cost_funs):
            res += cost_fn([kb, fc, ly, n])
        return res
    return cost_funs_sum

def perform_mh_fit(log_likelihood_fn, opts):
    n_sources = len(opts.sources)
    initial_pos = [0.5, 0.5, 0.5] + [1] * n_sources
    cov = np.diag([1e-5, 1e-5, 1e-5] + [5e-5] * n_sources)
    rng = np.random.default_rng(opts.seed)
    par_names = ['kb', 'fc', 'ly'] + list(opts.sources)

    sampler = SamplerMH(log_likelihood_fn, initial_pos, cov, par_names, rng)
    sampler.estimate_covariance(10, 1000, 10000)
    sampler.sample(opts.n_samples)
    outname = f"{opts.model}-mh-{'-'.join(opts.sources)}-{opts.dataset}-{opts.file_number}"
    sampler.save(opts.output, outname, metadata=vars(opts))
    return sampler

def perform_ultranest_fit(log_likelihood_fn, opts):
    def prior_transform(cube):
        params = cube.copy()
        n_lo, n_hi = 0.8, 1.2
        for i in range(3, len(cube)):
            params[i] = cube[i] * (n_hi - n_lo) + n_lo
        return params
    par_names = ['kb', 'fc', 'ly'] + list(opts.sources)

    sampler = ReactiveNestedSampler(par_names, log_likelihood_fn, prior_transform)
    result = sampler.run(
            viz_callback=False,
            show_status=False,
            )
    # result['metadata'] = vars(opts)
    outname = f"{opts.model}-ultranest-{'-'.join(opts.sources)}-{opts.dataset}-{opts.file_number}"
    with lzma_open(f'{opts.output}/{outname}.xz', 'wb') as file:
        pickle_dump(result, file)
    return result

def perform_minuit_fit(chi2, par_init, opts, cholesky_cov=None):
    m = Minuit(chi2, par_init)
    # m.precision = 1e-7
    # m.print_level = 3
    par_edges = [(0, 56), (0, 220), (0, 2506)] + [(0.9, 1.1)]*len(list(opts.sources))
    par_names = ['kb', 'fc', 'ly'] + list(opts.sources)

    # Set edges, make fit, unset edges, make fit
    if par_edges:
        for par, edge in zip(m.parameters, par_edges):
            m.limits[par] = edge
        m.migrad(ncall=100000)
        for p_name in m.pos2var:
            m.limits[p_name] = (None, None)
    m.migrad(ncall=100000)
    m.hesse()
    m.migrad(ncall=100000)
    m.minos()

    # Save results
    result = dict()
    result['valid'] = m.valid
    result['message'] = str(m.fmin)
    result['fun'] = m.fval
    result['covariance'] = np.array(m.covariance)
    result['corr'] = np.array(m.covariance.correlation())
    if cholesky_cov is not None:
        result['cholesky_cov'] = cholesky_cov

    xdict, errorsdict, profiles_dict = dict(), dict(), defaultdict(dict)
    for m_name, phys_name in zip(m.parameters[:3], par_names):
        par = m.params[m_name]
        xdict[phys_name] = par.value
        errorsdict[phys_name] = par.error

        a, fa, ok = m.mnprofile(m_name, subtract_min=True)
        profiles_dict[phys_name]['x'] = a
        profiles_dict[phys_name]['fcn'] = fa
        profiles_dict[phys_name]['valid'] = ok
        profiles_dict[phys_name]['merror'] = par.merror
        profiles_dict[phys_name]['merror_valid'] = m.merrors[m_name].is_valid

    result['xdict'] = xdict
    result['errorsdict'] = errorsdict
    result['profiles_dict'] = profiles_dict

    outname = f"{opts.model}-iminuit-{'-'.join(opts.sources)}-{opts.dataset}-{opts.file_number}"
    with lzma_open(f'{opts.output}/{outname}.xz', 'wb') as file:
        pickle_dump(result, file)
    return m

def perform_fc_fit(chi2, par_init, par_true, opts, cholesky_cov=None):
    m = Minuit(chi2, par_init)
    # m.precision = 1e-7
    par_edges = [(0, 56), (0, 220), (0, 2506)] # + [(0.7, 1.3)]*len(list(opts.sources))
    par_names = ['kb', 'fc', 'ly'] + list(opts.sources)
    if par_edges:
        for par, edge in zip(m.parameters, par_edges):
            m.limits[par] = edge
        m.migrad()
        for p_name in m.pos2var:
            m.limits[p_name] = (None, None)
    m.migrad()
    res_best = (m.valid, m.fmin.fval, m.values.to_dict())

    res_dict = dict()
    res_dict['res_best'] = res_best
    for i, par in enumerate(('x0', 'x1', 'x2')):
        m.fixto(par, par_true[i])
        m.migrad()
        res_dict[f'res_true_{par}'] = (m.valid, m.fmin.fval, m.values.to_dict())
        m.fixed[par] = False
        for p in ('x0', 'x1', 'x2',):
            m.values[p] = res_best[-1][p]
        # res_true = (m.valid, m.fmin.fval, m.values.to_dict())

    outname = f"{opts.model}-FC-{'-'.join(opts.sources)}-{opts.dataset}-{opts.file_number}"
    with lzma_open(f'{opts.output}/{outname}.xz', 'wb') as file:
        pickle_dump(res_dict, file)
    return m

def read_data_eos(common_eos_path, dataset, file_n, sources):
    edges = data_configs['kNPE_bins_edges']
    data_dict = dict()
    for source_name in sources:
        rootpath = \
                f'root://eos.jinr.ru//{common_eos_path}/{source_name}/{dataset}/reco/reco-{file_n}.root'
        print(rootpath)
        with open_root(rootpath) as rootfile:
            energy = 1.07 * rootfile['TRec']['m_NPE'].array() / 1000.
            data, _ = np.histogram(energy, bins=edges)
            data_dict[source_name] = data
    return data_dict

def main(opts):
    # data_dict = read_data(opts.data, opts.sources)
    data_dict = read_data_eos(opts.common_eos_path, opts.dataset, opts.file_number, opts.sources)
    cholesky_cov = None # np.load('average_cov_cholesky_decomposed.npy')

    log_likelihoods = list()
    chi2s = list()
    for source in opts.sources:
        data = data_dict[source]
        model = Model(source, np.sum(data), model_type=opts.model, path_to_models=opts.model_path)
        log_likelihoods.append(LogLikelihood(data, model))
        if cholesky_cov is not None:
            model = Model(source, np.sum(data),
                          path_to_models=opts.model_path, cholesky_cov=cholesky_cov)
        chi2s.append(LogLikelihoodRatio(data, model))
    log_likelihood_sum = cost_funs_sum_wrapper(log_likelihoods)
    chi2_like = cost_funs_sum_wrapper(chi2s)

    if 'metropolis_hastings' in opts.fit_tool:
        sampler = perform_mh_fit(log_likelihood_sum, opts)
    if 'ultranest' in opts.fit_tool:
        result = perform_ultranest_fit(log_likelihood_sum, opts)
    if 'iminuit' in opts.fit_tool:
        ls_pars_init = [0.5]*3 if cholesky_cov is None else list(solve_triangular(cholesky_cov, [0.5, 0.5, 0.5], lower=True))
        init_pars = list(ls_pars_init) + [1]*len(list(opts.sources))

        m = perform_minuit_fit(chi2_like, init_pars, opts, cholesky_cov)
    if 'FC' in opts.fit_tool:
        ls_pars_init = [0.5]*3 if cholesky_cov is None else list(solve_triangular(cholesky_cov, [0.5, 0.5, 0.5], lower=True))
        init_pars = list(ls_pars_init) + [1]*len(list(opts.sources))
        ls_pars_true = [0.525]*3 if cholesky_cov is None else list(solve_triangular(cholesky_cov, [0.525, 0.525, 0.525], lower=True))
        true_pars = list(ls_pars_true) + [1]*len(list(opts.sources))

        m = perform_fc_fit(chi2_like, init_pars, true_pars, opts, cholesky_cov)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--n-samples', type=int, default=300000, help='Number of samples to produce')
    parser.add_argument('--sources', required=True,
                        choices=sources_all,
                        nargs='+', help='Which sources to use')
    parser.add_argument('--fit-tool', nargs='+',
                        choices=('metropolis_hastings', 'ultranest', 'iminuit', 'FC',),
                        default='metropolis_hastings', help='What tools to use for fitting')

    parser.add_argument('--model', nargs='+',
                        choices=('tede', 'nfde',),
                        default='tede', help='What model to use')

    parser.add_argument('-d', '--data', default='./data', help='Path to find input data files')
    parser.add_argument('--common-eos-path',
                        default='/eos/juno/users/d/dolzhikov/neuromct_inputs',
                        help='Common path for all sources in eos')
    parser.add_argument('--dataset', default='testing_data', help='Dataset to use for fits')
    parser.add_argument('--file-number', default=0, type=int, help='File number to use')

    parser.add_argument('-o', '--output', default='./output', help='Path to store the output')
    parser.add_argument('-mpath','--model-path', default=None, help='Path to models')
    parser.add_argument('-s', '--seed', default=None, type=int, help='Seed to use for pseudorandom')
    opts = parser.parse_args()
    opts.sources = sort_sources_by_number(opts.sources)
    main(opts)
