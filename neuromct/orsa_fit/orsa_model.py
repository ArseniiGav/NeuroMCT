from dataclasses import dataclass
import orsa_fitter as orsa 
import pickle
import numpy as np
import torch

from orsa_fitter.model import DetectorParameter, NormalizationParameter
from ..models.ml.models_setup import setup
from ..configs import data_configs
from scipy.signal import savgol_filter
import copy
import json


@dataclass
class Conditions:
    kB: float
    fC: float
    LY: float

class ModelMCT:
    def __init__(self, source, model_type, device, parameters = [], n_samples = 1000, conditions = Conditions(0,0,0)):
        if isinstance(parameters, dict):
            self.parameters = parameters
        elif isinstance(parameters, np.ndarray) or isinstance(parameters, list):
            self.parameters = dict()
            for parameter in parameters:
                self.parameters[parameter.label] = parameter
        
        self.model_type = model_type
        self.model = setup(model_type, device)
        self.device = device

        if model_type == 'tede':
            self.n_samples = n_samples
        elif model_type == 'nfde':
            self.n_samples = n_samples
        else:
            raise Exception('Choose between TEDE and NFDE!')
        
        self.bins = data_configs['kNPE_bins_edges']
        self.centers = (self.bins[1:] + self.bins[:-1]) / 2
        self.conditions = conditions
        self.set_source(source)
        self.E_fit_min = 0.0
        self.E_fit_max = 16.0

        self.rebin = 1
        self.use_gpu = False
        self.use_shape_uncertainty = False
        self.resolution = self.bins[1]-self.bins[0]


    def set_source(self, source):
        
        self.source_ = source
        if source == 'Cs137':
            self.source_mask = np.array([0])
        elif source == 'K40':
            self.source_mask = np.array([1])
        elif source == 'Co60':
            self.source_mask = np.array([2])
        elif source == 'AmBe':
            self.source_mask = np.array([3])
        elif source == 'AmC':
            self.source_mask = np.array([4])
        else:
            raise Exception('Choose between Cs137, K40, Co60, AmBe, AmC!')
        
    def get_source(self):
        return self.source_
    
    
    source = property(get_source, set_source)

    def add_parameter(self, parameter):
        if parameter.label in self.parameters:
            raise Exception(f"Parameter {parameter.label} already present!")
        else:
            self.parameters[parameter.label] = parameter

        self.__init__(
            source = self.source,
            model_type = self.model_type,
            parameters=self.parameters,
            n_samples=self.n_samples,
            device=self.device
        )


    def remove_parameter(self, label):
        out_params = []
        for el in self.parameters:
            if label == el:
                pass
            else:
                out_params.append(self.parameters[el])

        self.__init__(
            parameters=self.out_params,
            n_samples=self.n_samples
        )


    def build_model(self, conditions_obj):
        params = np.array([[conditions_obj.kB, conditions_obj.fC, conditions_obj.LY]])
        source_types = np.array([self.source_mask])
        params = torch.tensor(params, dtype=torch.float64)
        source_types = torch.tensor(source_types, dtype=torch.int64)
        if self.model_type == 'tede':
            pred_spectra = self.model(params.to(self.device), source_types.to(self.device))
            counts = pred_spectra.cpu().detach().numpy().squeeze(0)
            return orsa.spectrum.ReconstructedSpectrum(E = self.centers, counts = counts, isPDF = False).norm()
        elif self.model_type == 'nf':
            conditions = np.array([[conditions_obj.kB, conditions_obj.fC, conditions_obj.LY]])
            conditions = torch.Tensor(conditions).to(self.device)
            prob_per_x_centers = torch.exp(self.model.log_prob(torch.Tensor(self.centers / 1000.).unsqueeze(1).to(self.device), conditions, self.source_)).flatten().cpu().detach().numpy()
            return orsa.spectrum.ReconstructedSpectrum(E = self.centers, counts = prob_per_x_centers + 1e-6, isPDF = False).norm()

    def get_ll(self, conditions_obj, E, prob_norm):
        conditions = np.array([[conditions_obj.kB, conditions_obj.fC, conditions_obj.LY]])
        conditions = torch.Tensor(conditions).to(self.device)
        ll_array = self.model.log_prob((torch.Tensor(E).to(self.device) / 1000.).unsqueeze(1).to(self.device), conditions, self.source_).flatten().cpu().detach().numpy() / prob_norm
        return ll_array

    def get_spectrum(self, asimov=False):
        spectrum = (
            self.calc_spectra()['ReconstructedSpectrum']
            .to_hist()
        )
        if asimov:
            return spectrum
        else:
            return spectrum.run_pseudoexperiment()

    def calc_spectra(self, vis2rec=True, getEcomp=False, kind="spectra"):
        
        norm = 0
        found = False
        for el in self.parameters:
            if isinstance(self.parameters[el], DetectorParameter):
                setattr(self.conditions, el, self.parameters[el].value)
            elif isinstance(self.parameters[el], NormalizationParameter):
                if found: raise Exception('Only one source per model implemented!')
                found = True
                norm = self.parameters[el].value

            else:
                raise Exception(
                    f"I do not know what to do with parameter {el} of general class {type(self.parameters[el])}!"
                )
        
        return dict(ReconstructedSpectrum = norm*self.build_model(self.conditions))
    
    def get_priors(self):
        return dict()
    
    def update_values(self, values, labels):
        for label, value in zip(labels, values):
            self.parameters[label].value = value

    def update_errors(self, errors, labels):
        for label, error in zip(labels, errors):
            self.parameters[label].error = error

    def get_priors(self):
        prior_dict = dict()
        for param in self.parameters:
            if len(self.parameters[param].prior) > 0:
                prior_dict[param] = self.parameters[param].prior
        return prior_dict
    
    def copy(self):
        return copy.deepcopy(self)

    def to_dict(self):
        out = dict()
        to_store = ["source", "model_type", "parameters", "n_samples", "conditions"]
        
        for a in to_store:
            if a == "conditions":
                out[a] = getattr(self, a).__dict__
            elif a == "parameters":
                out["parameters"] = dict()
                for param in self.parameters:
                    out["parameters"][param] = self.parameters[param].to_dict()
            else:
                out[a] = getattr(self, a)
        return out

    def to_json(self, filename):
        to_json(self, filename)



class NumpyEncoder(json.JSONEncoder):
    """
    Special json encoder for numpy types
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class NumpyDecoder(json.JSONDecoder):
    """
    Custom JSON decoder class for decoding NumPy ndarrays from lists within dictionaries.
    """

    def decode(self, s):
        def convert_to_array(obj):
            if isinstance(obj, list):
                return np.array(obj)
            return obj

        obj = json.JSONDecoder.decode(self, s)
        # Recursively convert lists to arrays within dictionaries
        if isinstance(obj, dict):
            obj = {key: convert_to_array(value) for key, value in obj.items()}
        return obj


def to_json(self, filename):
    with open(filename, "w") as outfile:
        json.dump(self.to_dict(), outfile, indent=4, cls=NumpyEncoder)


def from_json(filename):
    with open(filename) as json_file:
        data = json.load(json_file, cls=NumpyDecoder)

    return data