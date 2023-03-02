import numpy as np
from egenerator.utils.configurator import ManagerConfigurator
from matplotlib import pyplot as plt


# load and build model
# model_dir: path to an exported event-generator model
# example: 1-cascade model
model_dir = (
        './exported_models/starter_cascade_7param_muonl3_21220/'
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
params = [[0., 0., 0, np.deg2rad(42), np.deg2rad(330), 10000, 0]]

# run TF and get model expectation
# Note: running this the first time will trace the model.
# Consecutive calls will be faster
result_tensors = get_dom_expectation(params)

# get PDF and CDF values for some given times x
# these have shape: [n_batch, 86, 60, len(x)]
x = np.linspace(0., 3500, 1000)
pdf_values = model.pdf(x, result_tensors=result_tensors)
cdf_values = model.cdf(x, result_tensors=result_tensors)

# let's plot the PDF at DOMs 25 through 35 of String 36:
fig, ax = plt.subplots()
batch_id = 0  # we only injected one cascade via `params`
string = 36
for om in range(25, 35):
    ax.plot(
        x, pdf_values[batch_id, string - 1, om - 1],
        label='DOM: {:02d} | String {:02d}'.format(om, string),
    )
ax.legend()
ax.set_xlabel('Time / ns')
ax.set_ylabel('Density')


# ---------------------
# sweep through zen/azi
# ---------------------

string = 1
om = 1
for dzen in np.linspace(0, 180, 5):
    for azi in np.linspace(0, 360, 5):
        for energy in [1, 10, 100, 1000, 10000]:
            params = [
                [-256.1400146484375, -521.0800170898438, 480.,
                 np.radians(180-dzen), np.radians(azi), energy, 0]]
            result_tensors = get_dom_expectation(params)
            print('E: {} | PE: {}'.format(
                energy,
                result_tensors['dom_charges'][0, string - 1, om - 1, 0]))


# # cascade only
# model_dir_cascade = (
#     '/data/ana/PointSource/DNNCascade/utils/exported_models/version-0.0/'
#     'egenerator/cascade_7param_noise_tw_BFRv1Spice321_01/models_0000/cascade'
# )
# configurator_cscd = ManagerConfigurator(model_dir_cascade)


# cascade only
result_tensors_cscd = result_tensors['nested_results']['cascade']
model_cscd = model.sub_components['cascade']
pdf_values_cscd = model_cscd.pdf(x, result_tensors=result_tensors_cscd)

charges_cscd = result_tensors_cscd['dom_charges']

result_tensors_noise = result_tensors['nested_results']['noise']
charges_noise = result_tensors_noise['dom_charges']
