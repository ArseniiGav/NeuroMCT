from dataclasses import dataclass
import orsa_fitter as orsa 
import pickle
import numpy as np
import torch

from orsa.model import DetectorParameter, NormalizationParameter
from model_setup import setup
from scipy.signal import savgol_filter
import copy
import json


@dataclass
class Conditions:
    kB: float
    fC: float
    LY: float

class ModelMCT:
    def __init__(self, source, model_type, model_path, parameters = [], n_samples = 1000, conditions = Conditions(0,0,0)):
        # TODO: change it for different models, we should build a parser function
        if isinstance(parameters, dict):
            self.parameters = parameters
        elif isinstance(parameters, np.ndarray) or isinstance(parameters, list):
            self.parameters = dict()
            for parameter in parameters:
                self.parameters[parameter.label] = parameter
        
        self.model_type = model_type
        self.model_path = model_path
        self.ml_generator = setup(model_type, model_path)
        self.device = "cpu"#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if model_type == 'gan':
            self.n_samples = n_samples
        elif model_type == 'transformer':
            self.n_samples = n_samples
        elif model_type == 'regressor':
            self.n_samples = n_samples
        elif model_type == 'nf':
            self.n_samples = n_samples
        else:
            raise Exception('Choose between gan and transformer!')
        
        self.latent_dim = 20
        self.num_conditions = 8
        self.bins = np.arange(400, 16401, 20)
        self.centers = (self.bins[1:] + self.bins[:-1]) / 2
        self.conditions = conditions
        self.set_source(source)
        self.E_fit_min = 400
        self.E_fit_max = 16400

        self.rebin = 1
        self.use_gpu = False
        self.use_shape_uncertainty = False
        self.resolution = self.bins[1]-self.bins[0]


    def set_source(self, source):
        
        self.source_ = source
        if source == 'Co60':
            self.source_mask = np.array([0,0,1,0,0])
        elif source == 'K40':
            self.source_mask = np.array([0,1,0,0,0])
        elif source == 'Cs137':
            self.source_mask = np.array([1,0,0,0,0])
        elif source == 'AmBe':
            self.source_mask = np.array([0,0,0,1,0])
        elif source == 'AmC':
            self.source_mask = np.array([0,0,0,0,1])
        else:
            raise Exception('Choose between Co60, K40, Cs137, AmBe, AmC!')
        
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
            model_path = self.model_path,
            parameters=self.parameters,
            n_samples=self.n_samples
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
        conditions = np.array([
            [conditions_obj.kB, conditions_obj.fC, conditions_obj.LY,
             self.source_mask[0], self.source_mask[1], self.source_mask[2], self.source_mask[3], self.source_mask[4]]
        ])
        conditions = torch.Tensor(conditions)
        condition = (torch.ones((self.n_samples, self.num_conditions))*conditions).to(self.device)#.type_as(noise)
        if self.model_type == 'gan':
            noise = torch.randn(self.n_samples, self.latent_dim).to(self.device)
            counts = self.ml_generator(noise, condition).cpu().detach().numpy().mean(0)
            return orsa.spectrum.ReconstructedSpectrum(E = self.centers, counts = counts, isPDF = False).norm()
        elif self.model_type == 'regressor':
            pred_spectra = self.ml_generator(condition)
            counts = pred_spectra.cpu().detach().numpy().mean(0)
            return orsa.spectrum.ReconstructedSpectrum(E = self.centers, counts = counts, isPDF = False).norm()
        elif self.model_type == 'transformer':
            pred_spectra = self.ml_generator(condition)
            counts = pred_spectra.cpu().detach().numpy().mean(0)
            return orsa.spectrum.ReconstructedSpectrum(E = self.centers, counts = counts, isPDF = False).norm()
        elif self.model_type == 'nf':
            conditions = np.array([[conditions_obj.kB, conditions_obj.fC, conditions_obj.LY]])
            conditions = torch.Tensor(conditions).to(self.device)
            prob_per_x_centers = torch.exp(self.ml_generator.log_prob(torch.Tensor(self.centers / 1000.).unsqueeze(1).to(self.device), conditions, self.source_)).flatten().cpu().detach().numpy()
            return orsa.spectrum.ReconstructedSpectrum(E = self.centers, counts = prob_per_x_centers + 1e-6, isPDF = False).norm()

    def get_ll(self, conditions_obj, E, prob_norm):
        conditions = np.array([[conditions_obj.kB, conditions_obj.fC, conditions_obj.LY]])
        conditions = torch.Tensor(conditions).to(self.device)
        ll_array = self.ml_generator.log_prob((torch.Tensor(E).to(self.device) / 1000.).unsqueeze(1).to(self.device), conditions, self.source_).flatten().cpu().detach().numpy() / prob_norm
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