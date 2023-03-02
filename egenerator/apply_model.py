'''
Program to apply a trained and exported model to events for reconstruction.
Makes use of the I3TraySegments ApplyEventGeneratorReconstruction and 
ApplyEventGeneratorVisualizeBestFit.
'''
import os
import click
from I3Tray import I3Tray
from icecube import icetray, dataio, hdfwriter
from icecube.weighting import get_weighted_primary
from egenerator.ic3.segments import ApplyEventGeneratorReconstruction, ApplyEventGeneratorVisualizeBestFit
from modules.utils.recreate_and_add_mmc_tracklist import RerunProposal
from ic3_labels.labels.modules import MCLabelsCascades, MCLabelsMuonScattering, MCLabelsDeepLearning

import numpy as np
from egenerator.utils.configurator import ManagerConfigurator
from matplotlib import pyplot as plt

# Insert command line arguments
@click.command()
@click.argument('input_file_pattern', type=click.Path(exists=True),
                required=True, nargs=-1)
@click.option('-o', '--outfile', default='egen_output',
              help='Name of output file without file ending.')
@click.option('-m', '--model_names',
              default='getting_started_model',
              help='Parent directory of exported models.')
@click.option('-d', '--models_dir',
              default='{EGEN_HOME}/exported_models',
              help='Parent directory of exported models.')
@click.option('-g', '--gcd_file',
              default='/cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_IC86_Merged.i3.gz',
              help='GCD File to use.')
@click.option('-j', '--num_cpus',
              default=8,
              help='Number of CPUs to use if run on CPU instead of GPU')
@click.option('--i3/--no-i3', default=True)
@click.option('--hdf5/--no-hdf5', default=True)
@click.option('-p', '--plot', default=False, 
                help='Add visualization of reconstruction resutls')

def main(input_file_pattern, outfile, model_names, models_dir, gcd_file, 
        num_cpus, i3, hdf5, plot):

    # create output directory if necessary
    base_path = os.path.dirname(outfile)
    if not os.path.isdir(base_path):
        print('\nCreating directory: {}\n'.format(base_path))
        os.makedirs(base_path)

    # expand models_dir with environment variable
    models_dir = models_dir.format(EGEN_HOME=os.environ['EGEN_HOME'])
    
    HDF_keys = ['LabelsDeepLearning', 'masked_pulses', 'BrightDOMs', 'time_exclusion', 'MCPrimary', 
        'OnlineL2_PoleL2MPEFit_MuEx', 'OnlineL2_PoleL2MPEFit_TruncatedEnergy_AllBINS_Muon']

    tray = I3Tray()

    # read in files 
    file_name_list = [str(gcd_file)]
    file_name_list.extend(list(input_file_pattern))
    tray.AddModule('I3Reader', 'reader', Filenamelist=file_name_list)

    # add labels
    tray.AddModule(get_weighted_primary, 'getWeightedPrimary',
                  If=lambda f: not f.Has('MCPrimary'))
    # update mc trees to match that of create training data ... also compare rest of these with that file
    # tray.AddModule("Copy", "copy_mctrees", keys=['I3MCTree_preMuonProp', 'I3MCTree'])
    RerunProposal_kwargs = {}
    tray.AddSegment(
                RerunProposal, 'RerunProposal',
                mctree_name='I3MCTree',
                # Merging causes issues with MuonGunWeighter (needs investigation)
                # merge_trees=['BackgroundI3MCTree'],
                **RerunProposal_kwargs
                )
    tray.AddModule("Copy", "copy_series", keys=['I3MCPulseSeriesMap', 'I3MCPESeriesMap'])
    tray.AddModule(MCLabelsDeepLearning, 'MCLabelsDeepLearning',
                  PulseMapString='HVInIcePulses',
                  PrimaryKey='MCPrimary',
                  # ExtendBoundary=0.,
                  OutputKey='LabelsDeepLearning')

    # collect model and output names
    if isinstance(model_names, str):
        model_names = [str(model_names)]
    output_names = ['EventGenerator_{}'.format(m) for m in model_names]

    # make sure egen will be written to hdf5 file
    for outbox in output_names:
        if outbox not in HDF_keys:
            HDF_keys.append(outbox)
            HDF_keys.append(outbox +'_I3Particle')

    # Apply model
    # Use model for reconstruction of events
    tray.AddSegment(
        ApplyEventGeneratorReconstruction, 'ApplyEventGeneratorReconstruction',
        pulse_key='HVInIcePulses',
        dom_and_tw_exclusions=['BadDomsList', 'CalibrationErrata', 'SaturationWindows'],
        partial_exclusion=True,
        exclude_bright_doms=True,
        model_names=['starter_cascade_7param_muonl3_21220'],
        seed_keys=['MyAwesomeSeed'],
        model_base_dir='/data/i3home/sgray/egen/repositories/event-generator/exported_models/',
        )

    # Visualize reconstruction
    tray.AddSegment(
        ApplyEventGeneratorVisualizeBestFit, 'ApplyEventGeneratorVisualizeBestFit',
        pulse_key='HVInIcePulses',
        dom_and_tw_exclusions=['BadDomsList', 'CalibrationErrata', 'SaturationWindows'],
        partial_exclusion=True,
        exclude_bright_doms=True,
        model_names=['starter_cascade_7param_muonl3_21220'],
        reco_key='LabelsDeepLearning',
        output_dir='./output/starter_cascade_7param_muonl3_21220/visualization/',
        )

    # Write output
    if i3:
        tray.AddModule("I3Writer", "EventWriter",
                       filename='{}.i3.bz2'.format(outfile))

    if hdf5:
        tray.AddSegment(hdfwriter.I3HDFWriter, 'hdf',
                        Output='{}.hdf5'.format(outfile),
                        CompressionLevel=9,
                        Keys=HDF_keys,
                        SubEventStreams=['InIceSplit'])
    tray.AddModule('TrashCan', 'YesWeCan')
    tray.Execute()

    if plot:

        # Load model and plot PDFs at certain DOMs

        # load and build model
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


if __name__ == '__main__':
    main()
