import multiprocessing
import pickle
from copy import deepcopy

import numpy as np
import uproot

import orsa_fitter as orsa

from neuromct.orsa_fit.orsa_model import Conditions, ModelMCT
from neuromct.configs import data_configs

import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def load_simulation(n, bins, path = "/mnt/arsenii/NeuroMCT/kB_fC_LY_10k_events/<el>/testing_data2_1/reco/reco-<n>.root"): #testing_data
    centers = (bins[1:] + bins[:-1]) / 2
    out = dict()

    for el in ['Co60', 'K40', 'Cs137', 'AmBe', 'AmC']:
        temp_path = path.replace('<el>', el).replace('<n>', str(n))
        reco_default = uproot.open(temp_path)
        energy = 1.07 * np.array(reco_default['TRec']['m_NPE'].array()) / 1000.
        counts, _ = np.histogram(energy, bins=bins)
        out[el] = orsa.spectrum.ReconstructedSpectrum(E=centers, counts = counts, isPDF=False) #hasXS = True

    return out

def get_conditions(n):

    kB_list = [15.45]
    fC_list = [0.525]
    LY_list = [10100]
    
    kB_list = np.array(kB_list).reshape(-1, 1)
    fC_list = np.array(fC_list).reshape(-1, 1)
    LY_list = np.array(LY_list).reshape(-1, 1)
    conditions = np.concatenate([kB_list, fC_list, LY_list], axis=1)

    scaler = pickle.load(open(f"{data_configs['path_to_models']}/minmax_scaler.pkl", "rb"))

    conditions_default_reshaped = np.array(conditions[n]).reshape(1, -1)
    conditions_default_scaled = scaler.transform(conditions_default_reshaped)

    condition = Conditions(*conditions_default_scaled[0])
    return condition

def run_fit(n_sim):

    bins = data_configs['kNPE_bins_edges']
    sniper_spectra = load_simulation(n_sim, bins)
    
    # # SNiPER
    data1 = sniper_spectra['Co60']
    data2 = sniper_spectra['K40']
    data3 = sniper_spectra['Cs137']
    data4 = sniper_spectra['AmBe']
    data5 = sniper_spectra['AmC']
    
    n1 = data1.counts.sum()
    n2 = data2.counts.sum()
    n3 = data3.counts.sum()
    n4 = data4.counts.sum()
    n5 = data5.counts.sum()
        
    n_samples = 1
    kind = 'tede'
    
    conditions_default = get_conditions(0)
    
    model1 = ModelMCT(source='Co60', model_type=kind, n_samples=n_samples)
    model1.add_parameter(orsa.model.DetectorParameter(label = 'kB', value = conditions_default.kB, group = '', error = np.inf,   prior = {'flat': {'left':0, 'right':1}},   formatted_label = r'$k_B$'))
    model1.add_parameter(orsa.model.DetectorParameter(label = 'fC', value = conditions_default.fC, group = '', error = np.inf,   prior = {'flat': {'left':0, 'right':1}},   formatted_label = r'$f_C$'))
    model1.add_parameter(orsa.model.DetectorParameter(label = 'LY', value = conditions_default.LY, group = '', error = np.inf,   prior = {'flat': {'left':0, 'right':1}},   formatted_label = r'$L.Y.$'))
    model1.add_parameter(orsa.model.NormalizationParameter(label = 'N',   value = n1, group = '1',    generator = orsa.generator.reactor('HM', True, True),   is_oscillated=False, has_duty=True,  error = np.inf,  formatted_label = r'$N_\mathrm{evts}$', prior = {'flat': {'left':n1-5*np.sqrt(n1), 'right':n1+5*np.sqrt(n1)}}))
    
    model2 = ModelMCT(source='K40', model_type=kind, n_samples=n_samples)
    model2.add_parameter(orsa.model.DetectorParameter(label = 'kB', value = conditions_default.kB, group = '', error = np.inf,   prior = {'flat': {'left':0, 'right':1}},   formatted_label = r'$k_B$'))
    model2.add_parameter(orsa.model.DetectorParameter(label = 'fC', value = conditions_default.fC, group = '', error = np.inf,   prior = {'flat': {'left':0, 'right':1}},   formatted_label = r'$f_C$'))
    model2.add_parameter(orsa.model.DetectorParameter(label = 'LY', value = conditions_default.LY, group = '', error = np.inf,   prior = {'flat': {'left':0, 'right':1}},   formatted_label = r'$L.Y.$'))
    model2.add_parameter(orsa.model.NormalizationParameter(label = 'N',   value = n2, group = '2',    generator = orsa.generator.reactor('HM', True, True),   is_oscillated=False, has_duty=True,  error = np.inf,  formatted_label = r'$N_\mathrm{evts}$', prior = {'flat': {'left':n2-5*np.sqrt(n2), 'right':n2+5*np.sqrt(n2)}}))
    
    model3 = ModelMCT(source='Cs137', model_type=kind, n_samples=n_samples)
    model3.add_parameter(orsa.model.DetectorParameter(label = 'kB', value = conditions_default.kB, group = '', error = np.inf,   prior = {'flat': {'left':0, 'right':1}},   formatted_label = r'$k_B$'))
    model3.add_parameter(orsa.model.DetectorParameter(label = 'fC', value = conditions_default.fC, group = '', error = np.inf,   prior = {'flat': {'left':0, 'right':1}},   formatted_label = r'$f_C$'))
    model3.add_parameter(orsa.model.DetectorParameter(label = 'LY', value = conditions_default.LY, group = '', error = np.inf,   prior = {'flat': {'left':0, 'right':1}},   formatted_label = r'$L.Y.$'))
    model3.add_parameter(orsa.model.NormalizationParameter(label = 'N',   value = n3, group = '3',    generator = orsa.generator.reactor('HM', True, True),   is_oscillated=False, has_duty=True,  error = np.inf,  formatted_label = r'$N_\mathrm{evts}$', prior = {'flat': {'left':n3-5*np.sqrt(n3), 'right':n3+5*np.sqrt(n3)}}))
    
    model4 = ModelMCT(source='AmBe', model_type=kind, n_samples=n_samples)
    model4.add_parameter(orsa.model.DetectorParameter(label = 'kB', value = conditions_default.kB, group = '', error = np.inf,   prior = {'flat': {'left':0, 'right':1}},   formatted_label = r'$k_B$'))
    model4.add_parameter(orsa.model.DetectorParameter(label = 'fC', value = conditions_default.fC, group = '', error = np.inf,   prior = {'flat': {'left':0, 'right':1}},   formatted_label = r'$f_C$'))
    model4.add_parameter(orsa.model.DetectorParameter(label = 'LY', value = conditions_default.LY, group = '', error = np.inf,   prior = {'flat': {'left':0, 'right':1}},   formatted_label = r'$L.Y.$'))
    model4.add_parameter(orsa.model.NormalizationParameter(label = 'N',   value = n4, group = '4',    generator = orsa.generator.reactor('HM', True, True),   is_oscillated=False, has_duty=True,  error = np.inf,  formatted_label = r'$N_\mathrm{evts}$', prior = {'flat': {'left':n4-5*np.sqrt(n4), 'right':n4+5*np.sqrt(n4)}}))
    
    model5 = ModelMCT(source='AmC', model_type=kind, n_samples=n_samples)
    model5.add_parameter(orsa.model.DetectorParameter(label = 'kB', value = conditions_default.kB, group = '', error = np.inf,   prior = {'flat': {'left':0, 'right':1}},   formatted_label = r'$k_B$'))
    model5.add_parameter(orsa.model.DetectorParameter(label = 'fC', value = conditions_default.fC, group = '', error = np.inf,   prior = {'flat': {'left':0, 'right':1}},   formatted_label = r'$f_C$'))
    model5.add_parameter(orsa.model.DetectorParameter(label = 'LY', value = conditions_default.LY, group = '', error = np.inf,   prior = {'flat': {'left':0, 'right':1}},   formatted_label = r'$L.Y.$'))
    model5.add_parameter(orsa.model.NormalizationParameter(label = 'N',   value = n5, group = '5',    generator = orsa.generator.reactor('HM', True, True),   is_oscillated=False, has_duty=True,  error = np.inf,  formatted_label = r'$N_\mathrm{evts}$', prior = {'flat': {'left':n5-5*np.sqrt(n5), 'right':n5+5*np.sqrt(n5)}}))
    
    true_values = np.array([conditions_default.kB, conditions_default.fC, conditions_default.LY, n1, n2, n3, n4, n5])
    
    for model in [model1, model2, model3, model4, model5]:
        for el in model.parameters:
            try:
                del model.parameters[el].prior['positive']
            except Exception as e:
                # print(e)
                pass
    
    cf1 = orsa.probability.CostFunction(ll=orsa.probability.ll_binned, data = data1, model = model1, ll_args=dict())
    cf2 = orsa.probability.CostFunction(ll=orsa.probability.ll_binned, data = data2, model = model2, ll_args=dict())
    cf3 = orsa.probability.CostFunction(ll=orsa.probability.ll_binned, data = data3, model = model3, ll_args=dict())
    cf4 = orsa.probability.CostFunction(ll=orsa.probability.ll_binned, data = data4, model = model4, ll_args=dict())
    cf5 = orsa.probability.CostFunction(ll=orsa.probability.ll_binned, data = data5, model = model5, ll_args=dict())
    cf = cf1+cf2+cf3+cf4+cf5
    
    samples = 2000
    result_mcmc = orsa.fit.emcee(cf, samples, err_scale=1e-2, err_scale_is_relative=True)
    result_mcmc.true_values = true_values
    
    scaler = pickle.load(open(f"{data_configs['path_to_models']}/minmax_scaler.pkl", "rb"))
    
    for i in range(16):
        result_mcmc.all_samples[i, :, :3] = scaler.inverse_transform(result_mcmc.all_samples[i, :, :3])
        
    result_mcmc.values[:3] = scaler.inverse_transform(result_mcmc.values[:3].reshape(1, -1))[0]
    result_mcmc.true_values[:3] = scaler.inverse_transform(result_mcmc.true_values[:3].reshape(1, -1))[0]
    
    discard = samples // 2
    result_mcmc.discard = discard
    
    orsa.utils.to_file(result_mcmc, f"/storage/jmct_paper/fit_results/tede/testing_data2_1/all_samples/all_samples_{n_sim}.pkl")
    
    results = orsa.fit.Results(values=result_mcmc.values, errors=result_mcmc.errors,
                               correlation=result_mcmc.correlation, true_values=result_mcmc.true_values)
    orsa.utils.to_file(results, f"/storage/jmct_paper/fit_results/tede/testing_data2_1/fit_outputs/fit_outputs_{n_sim}.pkl")

max_workers = 50
trials = 1000
if __name__ == '__main__':  
  pool = multiprocessing.Pool(max_workers)
  args = [[n_sim] for n_sim in range(trials)]

  pool.starmap(run_fit, args)
  pool.close()
  pool.join()
