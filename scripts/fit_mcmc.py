from lzma import open as lzma_open

from pickle import dump as pickle_dump
from torch import tensor, int32, float32, set_num_threads, set_num_interop_threads
from ultranest import ReactiveNestedSampler
from uproot import open as open_root
import numpy as np

from neuromct.configs import data_configs
from neuromct.fit import SamplerMH, LogLikelihood
from neuromct.models.ml import setup

set_num_threads(1)
set_num_interop_threads(1)

sources_all = ('cs137', 'k40', 'co60', 'ambe', 'amc',)
source_to_number = dict(
        [(source_name, source_number) for source_number, source_name in enumerate(sources_all)]
        )

def sort_sources_by_number(sources):
    return sorted(sources, key=lambda name: source_to_number[name])

def read_data(data_path, sources):
    edges = data_configs['kNPE_bins_edges']
    data_dict = dict()
    for source_name in sources:
        rootfile = open_root(f'{data_path}/{source_name}.root')
        energy = 1.07 * rootfile['TRec']['m_NPE'].array() / 1000.
        data, _ = np.histogram(energy, bins=edges)
        data_dict[source_name] = data
    return data_dict

class Model:
    def __init__(self, source, integral, model_type='tede', device='cpu', path_to_models=None):
        self._source_n = source_to_number[source]
        self._integral = integral
        self._model = setup(model_type, device, path_to_models)

    def __call__(self, pars):
        kb, fc, ly, n = pars
        nl_pars = tensor([kb, fc, ly], dtype=float32).unsqueeze(0)
        out = self._model(
                nl_pars, tensor([[self._source_n]], dtype=int32)
                ).detach().numpy()[0] * self._integral
        return n * out

def log_likelihood_sum_wrapper(likelihoods):
    def log_likelihood_sum(pars):
        res = 0
        kb, fc, ly = pars[:3]
        norm_consts = pars[3:]
        for (n, likelihood) in zip(norm_consts, likelihoods):
            res += likelihood([kb, fc, ly, n])
        return res
    return log_likelihood_sum

def perform_mh_fit(likelihood_fn, opts):
    n_sources = len(opts.sources)
    initial_pos = [0.5, 0.5, 0.5] + [1] * n_sources
    cov = np.diag([1e-5, 1e-5, 1e-5] + [5e-5] * n_sources)
    rng = np.random.default_rng(opts.seed)
    par_names = ['kb', 'fc', 'ly'] + list(opts.sources)

    sampler = SamplerMH(likelihood_fn, initial_pos, cov, par_names, rng)
    sampler.estimate_covariance(10, 1000, 10000)
    sampler.sample(opts.n_samples)
    outname = f"mcmc-mh-{'-'.join(opts.sources)}"
    sampler.save(opts.output, outname, metadata=vars(opts))

def perform_ultranest_fit(likelihood_fn, opts):
    def prior_transform(cube):
        params = cube.copy()
        n_lo, n_hi = 0.8, 1.2
        for i in range(3, len(cube)):
            params[i] = cube[i] * (n_hi - n_lo) + n_lo
        return params
    par_names = ['kb', 'fc', 'ly'] + list(opts.sources)

    sampler = ReactiveNestedSampler(par_names, likelihood_fn, prior_transform)
    result = sampler.run(
            viz_callback=False,
            show_status=False,
            )
    result['metadata'] = vars(opts)
    outname = f"ultranest-{'-'.join(opts.sources)}"
    with lzma_open(f'{opts.output}/{outname}.xz', 'wb') as file:
        pickle_dump(result, file)

def main(opts):
    data_dict = read_data(opts.data, opts.sources)
    log_likelihoods = list()
    for source in opts.sources:
        model = Model(source, np.sum(data_dict[source]), path_to_models=opts.model_path)
        log_likelihoods.append(LogLikelihood(data_dict[source], model))
    log_likelihood_sum = log_likelihood_sum_wrapper(log_likelihoods)

    if opts.sampler == 'metropolis_hastings':
        perform_mh_fit(log_likelihood_sum, opts)
    else:
        perform_ultranest_fit(log_likelihood_sum, opts)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--n-samples', type=int, default=500000, help='Number of samples to produce')
    parser.add_argument('--sources', required=True,
                        choices=sources_all,
                        nargs='+', help='Which sources to use')
    parser.add_argument('--sampler', choices=('metropolis_hastings', 'ultranest',),
                        default='metropolis_hastings', help='What sampler to use')
    parser.add_argument('-d', '--data', default='./data', help='Path to find input data files')
    parser.add_argument('-o', '--output', default='./output', help='Path to store the resulting chain')
    parser.add_argument('-mpath','--model-path', default=None, help='Path to models')
    parser.add_argument('-s', '--seed', default=None, type=int, help='Seed to use for pseudorandom')
    opts = parser.parse_args()
    opts.sources = sort_sources_by_number(opts.sources)
    main(opts)
