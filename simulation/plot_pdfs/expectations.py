import numpy as np
from egenerator.utils.configurator import ManagerConfigurator
from matplotlib import pyplot as plt


# load and build model
# model_dir: path to an exported event-generator model
# example: 1-cascade model
#model_dir = (
#   '/scratch/tmp/fvaracar/event_generator_outputs/exported_models/mcpe_times/'
#)


model_dir = (
    '/data/user/jvara/exported_models/event-generator/lom/lom_pulses_negative_june_2023'
)




configurator = ManagerConfigurator(model_dir)
manager = configurator.manager
model = configurator.manager.models[0]

# get function from model (this builds the graph internally)
get_dom_expectation = manager.get_model_tensors_function()

# --------------
# example usage:
# --------------

# define parameters of model
# The names and order of these are available via `model.parameter_names`
# In this case it is: [x, y, z, zenith, azimuth, energy, time]
# Well inject one cascade at (0, 0, 0) with energy of 10 TeV

params = [[-1425.37, 805.519, -242.513, 0.114939, 5.65254, 41924, 0]]

# run TF and get model expectation
# Note: running this the first time will trace the model.
# Consecutive calls will be faster
result_tensors = get_dom_expectation(params)

# get PDF and CDF values for some given times x
# these have shape: [n_batch, 86, 60, len(x)]
x = np.linspace(0., 20000., 1000)
pdf_values = model.pdf(x, result_tensors=result_tensors)
cdf_values = model.cdf(x, result_tensors=result_tensors)

#pdf_values_array = pdf_values.numpy()
#np.save("pdf_values.npy", pdf_values_array)

string = 1081
om = 55
pmt = 2
num_pmt = 16
pdf_values_array=pdf_values[0,string-1001,om*num_pmt+pmt]
#pdf_values_array = y.numpy()
np.save("pdf_values.npy", pdf_values_array)