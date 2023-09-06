import os
import tensorflow as tf
import numpy as np
from icecube import dataclasses, icetray
from icecube.icetray.i3logging import log_info, log_warn
from egenerator.utils.configurator import ManagerConfigurator
from egenerator.utils import basis_functions
import time
from scipy import stats

class goodness_of_fit(icetray.I3ConditionalModule):

    """
    This class evaluates the performance of trained Event-Generator models 
    using goodness-of-fit metrics. It is designed to work with a Poisson-like charge likelihood 
    and a mixed model of asymmetric Gaussians for time probabilities.

    For charge data, it uses a chi-square-like metric, C, calculated as (true_charge - predicted) / predicted_error.
      A perfect fit yields a "C" distribution with a mean of 0 and a standard deviation of 1.

    For time data, the class uses the Kolmogorov-Smirnov (KS) test or its variants. 
    This measures the largest gap between predicted and empirical cumulative distribution functions (CDFs), translating it into a p-value. 
    A perfect fit would yield a uniform distribution of p-values between 0 and 1, while a poor fit results in p-values near 0.

    The p-values or "C"s are provided in a RecoPulseSeriesMap, that can be read from an I3 file or written down to a HDF5 file.
    """

    def __init__(self,context):
        icetray.I3ConditionalModule.__init__(self, context)


        #Required settings
        self.AddParameter("cascade_key",
                          "The key of the cascade labels.",
                          "LabelsDeepLearning")
        self.AddParameter("pulse_key",
                          "The key of the pulses to evaluate",
                          "Gen2-mcpes")
        self.AddParameter("calculate_truth",
                          "Boolean. Whether to apply the KS test to sampled data"
                          "from the pdfs or not",
                          False)
        self.AddParameter("model_dir",
                          "Directory of the trained and exported model to use.",
                          "/data/user/jvara/exported_models/event-generator/lom/lom_mcpe_memory_september")
        self.AddParameter("num_pulse_threshold",
                          "Minimum number of pulses on a pulse series to evaluate the time goodness of fit.",
                          10)
        self.AddParameter("min_charge_pred",
                          "Most of predicted charges are 1e-7, since 0 is not possible. Avoid all these by setting a"
                          "charge threshold.",
                          0.05)
        

    def Configure(self):
        """Configures Module
        """

        self.cascade_key = self.GetParameter("cascade_key")
        self.pulse_key = self.GetParameter("pulse_key")
        self.calculate_truth = self.GetParameter("calculate_truth")
        self.model_dir = self.GetParameter("model_dir")
        self.num_pulse_threshold = self.GetParameter("num_pulse_threshold")
        self.min_charge_pred = self.GetParameter("min_charge_pred")
    
    
    def DAQ(self,frame):
        """Apply module to Q-frames
        """

        #Cascade lables
        cascade = frame[self.cascade_key]
        #Pulse (mcpe) labels
        pulses = frame[self.pulse_key]

        #Directory of the exported model to be evaluated
        model_dir = self.model_dir

        configurator = ManagerConfigurator(
                manager_dirs=[model_dir],
                num_threads=0,
            )

        manager = configurator.manager
        
        model = configurator.manager.models[0]

        params = [[cascade.pos.x, cascade.pos.y, cascade.pos.z, cascade.dir.zenith, cascade.dir.azimuth, cascade.energy, cascade.time]]

        get_dom_expectation = manager.get_model_tensors_function()
        
        #Obtain result tensors from the cascade-label parameters
        result_tensors = get_dom_expectation(params)

        #Define all I3RecoPulseSeriesMap to store p-values etc

        #ks test
        ks = dataclasses.I3RecoPulseSeriesMap()
        ks_series = dataclasses.I3RecoPulseSeries()

        #Charge goodness of fit
        charge_test = dataclasses.I3RecoPulseSeriesMap()
        charge_series = dataclasses.I3RecoPulseSeries()

        #ks test for sampled data
        ks_sampled = dataclasses.I3RecoPulseSeriesMap()
        ks_series_sampled= dataclasses.I3RecoPulseSeries()
        
        #Hardcoded for Gen2-dimensions, change in the future...
        all_charges = np.zeros([120,80,16])
        for om_key, pulse_list in pulses.items():
            
            cum_charge = 0
            for pulse in pulse_list:
                cum_charge+=pulse.npe
            
            all_charges[om_key.string-1001,om_key.om-1,om_key.pmt]=cum_charge

            #calculate predicted cdf
            def cdf(x):
                cdf_values = model.cdf(x,result_tensors=result_tensors)

                string = om_key.string-1001
                om = om_key.om-1
                pmt = om_key.pmt

                return cdf_values[0,string,(om-1)*16+pmt]

            #At least some pulses to have a "decent" empirical cdf
            if len(pulse_list)>=self.num_pulse_threshold:

                string = om_key.string - 1001
                om = om_key.om -1
                pmt = om_key.pmt

                #ks test for simulated data
                empirical_times = [pulse.time for pulse in pulse_list]
                empirical_times = np.array(sorted(empirical_times))
                #Compute goodness of fit
                result = stats.kstest(empirical_times, cdf)
                ks_pulse = dataclasses.I3RecoPulse()
                ks_pulse.time = len(pulse_list)
                ks_pulse.charge = result.pvalue
                ks_series.append(ks_pulse)
            
                if self.calculate_truth:
                    #Calculate ks for sampled data
                    sampled_times = self.sample_times(cascade=cascade,result_tensors=result_tensors,model=model,num_pulses=len(pulse_list),string=string,om=om,pmt=pmt)
                    sampled_times = sampled_times + cascade.time
                    result_sampled = stats.kstest(sampled_times, cdf)
                    ks_pulse_sampled = dataclasses.I3RecoPulse()
                    ks_pulse_sampled.time = len(pulse_list)
                    ks_pulse_sampled.charge = result_sampled.pvalue
                    ks_series_sampled.append(ks_pulse_sampled)

        #No minimum number of pulses threshold for the charge
        for string in range(120):
            for om in range(80):
                for pmt in range(16):
                    predicted_charge = result_tensors['dom_charges'][0,string, (om)*16 + pmt, 0]
                    

                    if predicted_charge>self.min_charge_pred:
                        
                        #Compute goodness of fit
                        x_value = (all_charges[string,om,pmt]-float(predicted_charge))/np.sqrt(float(predicted_charge))
                        charge_pulse = dataclasses.I3RecoPulse()
                        charge_pulse.charge = cascade.energy
                        charge_pulse.time = x_value
                        charge_series.append(charge_pulse)
        
        #Store everything on first pmt
        om_key = icetray.OMKey(0,0,0)
        
        ks[om_key]=ks_series
        charge_test[om_key]=charge_series
        frame['ks_test']=ks
        frame['charge_test']=charge_test

        if self.calculate_truth:
            ks_sampled[om_key]=ks_series_sampled
            frame['ks_sampled']=ks_sampled
        self.PushFrame(frame)
            

    def sample_times(self,cascade,result_tensors,model,num_pulses,string=0,om=0,pmt=0):
        """"
        Method to sample times from predicted pdfs. Same way as for simulations with event-generator.
        """
        model = model

        # get parameter names that have to be set
        param_names = sorted([n for n in model.parameter_names])

        # make sure that the only parameters that need to be set are provided
        included_parameters = [
                'azimuth', 'energy', 'time', 'x', 'y', 'z', 'zenith']

        # search if there is a common prefix that we can use
        prefix_list = []
        for name in included_parameters:
            if name in param_names:
                prefix_list.append('')
            else:
                found_match = False
                for param in param_names:
                    if param[-len(name):] == name:
                        # only allow prefixes ending on '_'
                        if param[-len(name)-1] == '_':
                            prefix_list.append(param[:-len(name)])
                            found_match = True
                if not found_match:
                    msg = (
                        'Did not find a parameter name match for "{}". Model '
                        'Parameter names are: {}'
                    ).format(name, param_names)
                    raise ValueError(msg)

        prefix_list = np.unique(prefix_list)
        if len(prefix_list) != 1:
            msg = 'Could not find common parameter prefix. Found: {}'.format(
                prefix_list)
            raise ValueError(msg)

        prefix = prefix_list[0]


        if 'latent_var_scale' not in result_tensors:
                result_tensors = result_tensors['nested_results'][
                    prefix[:-1]]

        cum_scale = np.cumsum(
            result_tensors['latent_var_scale'].numpy(), axis=-1)

        latent_var_mu = result_tensors['latent_var_mu'].numpy()
        latent_var_sigma = result_tensors['latent_var_sigma'].numpy()
        latent_var_r = result_tensors['latent_var_r'].numpy()
        cum_scale[..., -1] = 1.00000001
        current_time = int(time.time())
        random_service = np.random.RandomState(current_time)
        num_pe = num_pulses
        rngs = random_service.uniform(size=(num_pe, 2))
        num_pmt_om = 16
        idx = np.searchsorted(cum_scale[0, string, om*num_pmt_om + pmt], rngs[:, 0])
        pulse_mu = latent_var_mu[0, string, om*num_pmt_om + pmt, idx]
        pulse_sigma = latent_var_sigma[0, string, om*num_pmt_om + pmt, idx]
        pulse_r = latent_var_r[0, string, om*num_pmt_om + pmt, idx]
        
        pulse_times = basis_functions.asymmetric_gauss_ppf(
                            rngs[:, 1], mu=pulse_mu, sigma=pulse_sigma, r=pulse_r)

        pulse_times *= 1000.

        return pulse_times
