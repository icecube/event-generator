#!/usr/bin/env python
# encoding: utf-8

from __future__ import division
import os
import sys
sys.path.insert(1, '..')
import numpy as np
import getpass
import socket
import pickle

# FLOAT_PRECISION = tf.float32

def add_server_settings(settings):
    '''
        Adds server settings such as machine specific
        filepaths.

    Parameters
    ----------
    settings: dictionary
            The settings in the dictionary must at least contain:
                addWhichData: 'Which data to add'
                               one of: 'mix','summary','bins'
                # labelsToLoad: 'Monte Carlo Labels to load'
                #                one of: 'labels11','labels29','labels123','labels126'


    Returns
    -------
    settings: dictionray
            A dictionary with all machine specific settings added as well
            as standard settings that were not previously defined in the
            parameter settings

    '''
    # generate default settings if not given
    default = {
        'verbose' : True,
        'pathToPlots' : '',
        'model_path' : 'checkpoints/',
        'local_train_batch_size' : 1,
        'local_test_batch_size' : 1,
        'remote_train_batch_size' : 392,
        'remote_test_batch_size' : 392,
        'normalizationConstant': 1e-8,
        'PrimaryDirectionXKey' : 'PrimaryDirectionX',
        'PrimaryDirectionYKey' : 'PrimaryDirectionY',
        'PrimaryDirectionZKey' : 'PrimaryDirectionZ',
        'num_batch_splits' : None,
        'delay_batch_split_submission' : None,
        'dataset' : '11069',
        'labelsToLoad' : 'labels134',
        'addDOMCoordinates' : False,
        'model_name_suffix' : '',
        'use_relative_times' : True,
        }
    for key, value in default.items():
        if key not in settings.keys():
            settings[key] = value

    if settings['addWhichData'] == 'mix':
        numberOfBins = 12
        numberOfSummaryVars = 5
        outputFileAddition = ''
    elif settings['addWhichData'][:14] == 'CherenkovTrack':
        numberOfBins = 0
        numberOfSummaryVars = int(settings['addWhichData'][14:])
        outputFileAddition = 'cherenkovTrack{}/'.format(numberOfSummaryVars)
    # elif settings['addWhichData'] == 'mix600':
    #     numberOfBins = 600
    #     numberOfSummaryVars = 7
    #     outputFileAddition = 'mix600/'
    #     settings['num_batch_splits']  = 4
    elif settings['addWhichData'] == 'summary':
        numberOfBins = 0
        numberOfSummaryVars = 7
        outputFileAddition = 'summary/'
    elif settings['addWhichData'] == 'summaryV2':
        numberOfBins = 0
        numberOfSummaryVars = 9
        outputFileAddition = 'summaryV2/'
    elif settings['addWhichData'] == 'summaryV2_clipped':
        numberOfBins = 0
        numberOfSummaryVars = 9
        outputFileAddition = 'summaryV2_clipped/'
    elif settings['addWhichData'] == 'summaryV2_llh':
        numberOfBins = 0
        numberOfSummaryVars = 13
        outputFileAddition = 'summaryV2_llh/'
    elif settings['addWhichData'] == 'IntegratedCharge':
        numberOfBins = 0
        numberOfSummaryVars = 1
        outputFileAddition = 'IntegratedCharge/'
    # elif settings['addWhichData'] == 'bins':
    #     numberOfBins = 25
    #     numberOfSummaryVars = 0
    #     outputFileAddition = 'bins/'
    elif settings['addWhichData'] == 'bins25':
        numberOfBins = 25
        numberOfSummaryVars = 0
        outputFileAddition = 'bins025/'
    elif settings['addWhichData'] == 'bins100':
        numberOfBins = 100
        numberOfSummaryVars = 0
        outputFileAddition = 'bins100/'
    elif settings['addWhichData'] == 'bins100WF':
        numberOfBins = 100
        numberOfSummaryVars = 0
        outputFileAddition = 'bins100WF/'
    elif settings['addWhichData'] == 'llh20':
        numberOfBins = 0
        numberOfSummaryVars = 31
        outputFileAddition = 'llh20/'
    elif settings['addWhichData'] == 'llh':
        numberOfBins = 0
        numberOfSummaryVars = 5
        outputFileAddition = 'llh/'
    else:
        raise ValueError('Invalid value for addWhichData: {}'.format(settings['addWhichData']))
    # ---------------------------------------------------------------------

    # define if local or on madison cluster:
    local = False
    # if getpass.getuser() == 'mirco':
    #     local = True
    # elif getpass.getuser() == 'mhuennefeld':
    #     local = False
    # else:
    #     raise ValueError('User has to be mirco or mhuennefeld.')

    # data input files/ filelists
    if local:

        #------------------------------------
        #           Local
        #------------------------------------

        gcdfile = '/home/mirco/Repositories/Masterarbeit/Repository/deepLearning/data/GeoCalibDetectorStatus_2012.56063_V1.i3.gz'
        # file_val = '/media/mirco/B22E7EC12E7E7DE3/Users/Mirco/Desktop/datasets/11069/output/11069DOMPulseData00.hdf5' # first hundred files are for validation
        # file_val = '/media/mirco/B22E7EC12E7E7DE3/Users/Mirco/Desktop/datasets/11069/output/11069DOMPulseData65.hdf5' # first hundred files are for validation
        file_val = '../data/output/11069Benchmark.hdf5' # first hundred files are for validation
        # file_test = '/media/mirco/B22E7EC12E7E7DE3/Users/Mirco/Desktop/datasets/11069/output/11069DOMPulseData01.hdf5' # second hundred files are for testing

        if settings['dataset'] == '11069' and not 'pathToOutput' in settings:
            pathToOutput = '/media/mirco/B22E7EC12E7E7DE3/Users/Mirco/Desktop/datasets/11069/output/'
            # pathToInput = None # not defined here
            # inputFileBaseName = None # not defined here
            input_file_template = None


            # define train, test and validation dataset
            fileNumbers_val = range(0,100)
            fileNumbers_test = range(100,200)
            fileNumbers_train = range(200,201)

        elif settings['dataset'] == '11069_2017OnlineL2' and not 'pathToOutput' in settings:
            pathToOutput = '/media/mirco/B22E7EC12E7E7DE3/Users/Mirco/Desktop/datasets/11069/2017OnlineL2/output/'
            # pathToInput = '/media/mirco/B22E7EC12E7E7DE3/Users/Mirco/Desktop/datasets/11069/2017OnlineL2/'
            # inputFileBaseName = '2017OnlineL2_nugen_numu_IC86.2012.011069.0'
            input_file_template = '/media/mirco/B22E7EC12E7E7DE3/Users/Mirco/Desktop/datasets/11069/2017OnlineL2/{}/2017OnlineL2_nugen_numu_IC86.2012.011069.{:06d}.i3.bz2'

            # define train, test and validation dataset
            fileNumbers_val = range(0,100)
            fileNumbers_test = range(100,200)
            fileNumbers_train = range(200,201)


        elif settings['dataset'] == '11069_PS_final' and not 'pathToOutput' in settings:
            pathToOutput = '/media/mirco/B22E7EC12E7E7DE3/Users/Mirco/Desktop/datasets/11069_PS_final/output/'
            # pathToInput = None # not defined here
            # inputFileBaseName = None # not defined here
            input_file_template = None

            # define train, test and validation dataset
            fileNumbers_val = range(0,100)
            fileNumbers_test = range(100,200)
            fileNumbers_train = range(200,201)

        elif settings['dataset'] == 'NugenNormal' and not 'pathToOutput' in settings:
            pathToOutput = '/media/mirco/B22E7EC12E7E7DE3/Users/Mirco/Desktop/datasets/simulation/nugen_00001/normal/output/'
            # pathToInput = None
            # inputFileBaseName = None
            input_file_template = None
            gcdfile = '/media/mirco/B22E7EC12E7E7DE3/Users/Mirco/Desktop/datasets/simulation/nugen_00001/normal/GeoCalibDetectorStatus_2012.56063_V1.i3.gz'

            # define train, test and validation dataset
            fileNumbers_val = range(0,10)
            fileNumbers_test = range(10,20)
            fileNumbers_train = range(20,100)# + range(333,478) + range(666,790)

        elif settings['dataset'] == 'NugenPerfect' and not 'pathToOutput' in settings:
            pathToOutput = '/media/mirco/B22E7EC12E7E7DE3/Users/Mirco/Desktop/datasets/simulation/nugen_00001/perfect_hex_grid/output/'
            # pathToInput = None
            # inputFileBaseName = None
            input_file_template = None
            gcdfile = '/media/mirco/B22E7EC12E7E7DE3/Users/Mirco/Desktop/datasets/simulation/nugen_00001/perfect_hex_grid/GeoCalibDetectorStatus_2012.56063_V1_perfect_hex_grid.i3.gz'

            # define train, test and validation dataset
            fileNumbers_val = range(0,10)
            fileNumbers_test = range(10,20)
            fileNumbers_train = range(20,100)# + range(333,478) + range(666,790)

        # elif settings['dataset'] == 'NuE_IC86_2013_holeice_30_v4_l5':
        #     pathToOutput = '/home/mirco/Repositories/PHD/deeplearning/NuE/output/'
        #     # pathToInput = '/home/mirco/Repositories/PHD/deeplearning/NuE/'
        #     # inputFileBaseName = 'l5_00000001.i3.bz2'
        #     input_file_template = '/home/mirco/Repositories/PHD/deeplearning/NuE/data/{}/l5_{:08d}.i3.bz2'


        #     gcdfile = '/media/mirco/B22E7EC12E7E7DE3/Users/Mirco/Desktop/datasets/simulation/nugen_00001/perfect_hex_grid/GeoCalibDetectorStatus_2012.56063_V1_perfect_hex_grid.i3.gz'

        #     # define train, test and validation dataset
        #     fileNumbers_val = range(0,10)
        #     fileNumbers_test = range(10,20)
        #     fileNumbers_train = range(20,100)# + range(333,478) + range(666,790)

        else:
            for key in ['pathToOutput','input_file_template',
                        'gcdfile','file_number_range_val',
                        'file_number_range_test','file_number_range_train']:

                        if key not in settings.keys():
                            raise ValueError('Dataset {} is unknown. Key {} must be defined'.format(settings['dataset'], key))

            # todo clean this up, so this does not have to be done
            pathToOutput = settings['pathToOutput']
            del settings['pathToOutput']
            input_file_template = settings['input_file_template']
            gcdfile = settings['gcdfile']
            fileNumbers_val = range(*settings['file_number_range_val'])
            fileNumbers_test = range(*settings['file_number_range_test'])
            fileNumbers_train = range(*settings['file_number_range_train'])


        pathToOutput += outputFileAddition

        if 'cascade' == settings['dataset'][:7] or \
               settings['dataset'] in ['11069'] or \
               'Level2_muon_resimulation' in settings['dataset']:
            filelist_val = [pathToOutput+'{0:02d}000-{0:02d}999/{2}DOMPulseData_{1:08d}.hdf5'.format( int(n/1000), n, settings['dataset'])
                                for n in fileNumbers_val]

            filelist_test = [pathToOutput+'{0:02d}000-{0:02d}999/{2}DOMPulseData_{1:08d}.hdf5'.format(int(n/1000),n,settings['dataset'])
                                        for n in fileNumbers_test] * 200

            filelist_train = [pathToOutput+'{0:02d}000-{0:02d}999/{2}DOMPulseData_{1:08d}.hdf5'.format(int(n/1000),n,settings['dataset'])
                                    for n in fileNumbers_train] * 100 #100 epochs
        else:
            filelist_val = [pathToOutput+'{0:02d}000-{0:02d}999/{2}DOMPulseData_{1:04d}.hdf5'.format( int(n/1000), n, settings['dataset'])
                            for n in fileNumbers_val]

            filelist_test = [pathToOutput+'{0:02d}000-{0:02d}999/{2}DOMPulseData_{1:04d}.hdf5'.format(int(n/1000),n,settings['dataset'])
                                        for n in fileNumbers_test] * 200

            filelist_train = [pathToOutput+'{0:02d}000-{0:02d}999/{2}DOMPulseData_{1:04d}.hdf5'.format(int(n/1000),n,settings['dataset'])
                                        for n in fileNumbers_train] * 100 #100 epochs


        #------------------------------
        # Include additional datasets
        #------------------------------
        if 'add_datasets' in settings:
            assert len(settings['add_datasets']) == len(settings['add_pathToOutput'])
            assert len(settings['add_datasets']) == len(settings['add_file_number_range_val'])
            assert len(settings['add_datasets']) == len(settings['add_file_number_range_test'])
            assert len(settings['add_datasets']) == len(settings['add_file_number_range_train'])

            if 'dataset_weights' in settings:
                assert len(settings['add_datasets']) + 1 == len(settings['dataset_weights'])
                filelist_test = filelist_test * settings['dataset_weights'][0]
                filelist_train = filelist_train * settings['dataset_weights'][0]

            for d, dir_out, r_val, r_test, r_train, w in zip( settings['add_datasets'],
                                                           settings['add_pathToOutput'],
                                                           settings['add_file_number_range_val'],
                                                           settings['add_file_number_range_test'],
                                                           settings['add_file_number_range_train'],
                                                           settings['dataset_weights'][1:]
                                                            ):
                if 'cascade' == settings['dataset'][:7] or \
                       settings['dataset'] in ['11069'] or \
                       'Level2_muon_resimulation' in settings['dataset']:
                    files_val = [dir_out + outputFileAddition +'{0:02d}000-{0:02d}999/{2}DOMPulseData_{1:08d}.hdf5'.format( int(n/1000), n, d)
                                for n in range(*r_val)]

                    files_test = [dir_out + outputFileAddition +'{0:02d}000-{0:02d}999/{2}DOMPulseData_{1:08d}.hdf5'.format( int(n/1000), n, d)
                                for n in range(*r_test)] * 200 * w

                    files_train = [dir_out + outputFileAddition +'{0:02d}000-{0:02d}999/{2}DOMPulseData_{1:08d}.hdf5'.format( int(n/1000), n, d)
                                for n in range(*r_train)] * 200 * w
                else:
                    files_val = [dir_out + outputFileAddition +'{0:02d}000-{0:02d}999/{2}DOMPulseData_{1:04d}.hdf5'.format( int(n/1000), n, d)
                                for n in range(*r_val)]

                    files_test = [dir_out + outputFileAddition +'{0:02d}000-{0:02d}999/{2}DOMPulseData_{1:04d}.hdf5'.format( int(n/1000), n, d)
                                for n in range(*r_test)] * 200 * w

                    files_train = [dir_out + outputFileAddition +'{0:02d}000-{0:02d}999/{2}DOMPulseData_{1:04d}.hdf5'.format( int(n/1000), n, d)
                                for n in range(*r_train)] * 200 * w

                filelist_val.extend(files_val)
                filelist_test.extend(files_test)
                filelist_train.extend(files_train)
        #------------------------------

        # File list to create normModel
        if 'num_files_for_norm_model' in settings:
            num_files_for_norm_model = settings['num_files_for_norm_model']
        else:
            num_files_for_norm_model = 300

            if len(filelist_train) < 300:
                msg = '\033[93mReducing Number of files for creation of normalization model to {}\033[0m'
                print(msg.format(len(filelist_train)))
                num_files_for_norm_model = len(filelist_train)


        filelist_norm_train = np.random.choice(filelist_train, num_files_for_norm_model, replace=False)
        filelist_norm_val = np.random.choice(filelist_train, 50, replace=False)
        # fileNumbers = range(200,500)#range(200,500)
        # filelist_norm_train = [pathToOutput+'{0:02d}000-{0:02d}999/{2}DOMPulseData_{1:04d}.hdf5'.format(int(n/1000),n,settings['dataset'])
        #                             for n in fileNumbers]

        # fileNumbers = range(900,950)#range(900,950)
        # filelist_norm_val = [pathToOutput+'{0:02d}000-{0:02d}999/{2}DOMPulseData_{1:04d}.hdf5'.format(int(n/1000),n,settings['dataset'])
        #                             for n in fileNumbers]

        numOfThreads = 3
        train_batch_size = settings['local_train_batch_size']#2048
        test_batch_size = settings['local_test_batch_size']#2048

        # # # Test with single 700 files
        # file_test = pathToOutput+'/00000-00999/11069DOMPulseData_0000.hdf5'
        file_test = filelist_test[0]
        # # file_test = pathToOutput+'/00000-00999/11069DOMPulseData_0967.hdf5' # empty file
        # # file_val = '../data/output/11069DOMPulseData_0001.hdf5'
        # # filelist_train = np.tile(['../data/output/11069DOMPulseData_0000.hdf5'],100) # with 10*10 = 100 epochs

        # Test simulation files
        # file_test = '/media/mirco/B22E7EC12E7E7DE3/Users/Mirco/Desktop/datasets/simulation/NuGen/perfect_hex_grid/output/nugen_perfect_hex_grid_level2_processed_00000.hdf5'
        # file_test = '/media/mirco/B22E7EC12E7E7DE3/Users/Mirco/Desktop/datasets/simulation/NuGen/normal/output/nugen_level2_processed_00000.hdf5'
        # file_test = '/media/mirco/B22E7EC12E7E7DE3/Users/Mirco/Desktop/datasets/simulation/nugen_00001/normal/00000-00999/output/nugen_01_2017OnlineFilter_normal_00000.hdf5'
        # filelist_train = np.tile([file_test],100)
        # filelist_test = np.tile([file_test],100)

        # splines
        BareMuTimingSpline = '/home/mirco/i3_software/icerec/cvmfs_stuff/splines/InfBareMu_mie_prob_z20a10_V2.fits'
        BareMuAmplitudeSpline = '/home/mirco/i3_software/icerec/cvmfs_stuff/splines/InfBareMu_mie_abs_z20a10_V2.fits'

        # Tensorflow Session Config
        import tensorflow as tf
        config = tf.ConfigProto(gpu_options = tf.GPUOptions(
                                            #per_process_gpu_memory_fraction=0.5,
                                            allow_growth = True,
                                            ),
                                # intra_op_parallelism_threads=1,
                                # inter_op_parallelism_threads=1,
                                )

    else:
        if 'phobos' in socket.gethostname():

            #------------------------------------
            #           Phobos
            #------------------------------------
            # directory_prefix = '/fhgfs/users/mhuennefeld/Masterarbeit/'
            directory_prefix = '/net/big-tank/POOL/users/mhuennefeld/Masterarbeit/'

            # Get directories for dataset
            if settings['dataset'] == '11069' and not 'pathToOutput' in settings:
                pathToOutput = directory_prefix + 'data/11069/output/'
                # pathToInput = None # not defined here
                # inputFileBaseName = 'Level2_nugen_numu_IC86.2012.011069.0'
                input_file_template = None
                gcdfile = directory_prefix + 'data/11069/GeoCalibDetectorStatus_2012.56063_V1.i3.gz'

                # define train, test and validation dataset
                fileNumbers_val = range(0,100)
                fileNumbers_test = range(100,200)
                fileNumbers_train = range(200,6000)

            elif settings['dataset'] == '11069_2017OnlineL2' and not 'pathToOutput' in settings:
                pathToOutput = directory_prefix + 'data/11069/2017OnlineL2/output/'
                # pathToInput = None # not defined here
                # inputFileBaseName = '2017OnlineL2_nugen_numu_IC86.2012.011069.0'
                input_file_template = None
                gcdfile = directory_prefix + 'data/11069/GeoCalibDetectorStatus_2012.56063_V1.i3.gz'

                # define train, test and validation dataset
                fileNumbers_val = range(6000,10000)
                # fileNumbers_val = range(0,100)
                fileNumbers_test = range(100,200)
                fileNumbers_train = range(200,6000)

            elif settings['dataset'] == '11069_PS_final' and not 'pathToOutput' in settings:
                pathToOutput = directory_prefix + 'data/11069_PS_final/output/'
                # pathToInput = None # not defined here
                # inputFileBaseName = 'Final_v2_nugen_numu_IC86.2012.011069.0'
                input_file_template = None
                gcdfile = directory_prefix + 'data/11069/GeoCalibDetectorStatus_2012.56063_V1.i3.gz'

                # define train, test and validation dataset
                fileNumbers_val = range(6000,10000)
                # fileNumbers_val = range(0,100)
                fileNumbers_test = range(100,200)
                fileNumbers_train = range(200,6000)


            elif settings['dataset'] == 'NugenNormal' and not 'pathToOutput' in settings:
                pathToOutput = directory_prefix + 'data/nugen_00001/normal/output/'
                # pathToInput = '/fhgfs/users/mhuennefeld/simulation/data/nugen_00001/05_2017OnlineFilter/normal/'
                # inputFileBaseName = 'nugen_01_2017OnlineFilter_normal_'
                input_file_template = directory_prefix + 'data/nugen_00001/05_2017OnlineFilter/normal/{}/nugen_01_2017OnlineFilter_normal_{:05d}.i3.bz2'

                # pathToInput = '/fhgfs/users/mhuennefeld/simulation/data/nugen_00001/04_level2/normal/'
                # inputFileBaseName = 'nugen_01_level2_normal_'
                gcdfile = directory_prefix + 'data/nugen_00001/configuration/GeoCalibDetectorStatus_2012.56063_V1.i3.gz'

                # define train, test and validation dataset
                fileNumbers_val = range(0,10)
                fileNumbers_test = range(10,20)
                fileNumbers_train = range(20,151) + range(333,478) + range(666,790)

            elif settings['dataset'] == 'NugenPerfect' and not 'pathToOutput' in settings:
                pathToOutput = directory_prefix + 'data/nugen_00001/perfect_hex_grid/output/'
                # pathToInput = '/fhgfs/users/mhuennefeld/simulation/data/nugen_00001/05_2017OnlineFilter/perfect_hex_grid/'
                # inputFileBaseName = 'nugen_01_2017OnlineFilter_perfect_hex_grid_'
                input_file_template = directory_prefix + 'data/nugen_00001/05_2017OnlineFilter/perfect_hex_grid/{}/nugen_01_2017OnlineFilter_perfect_hex_grid_{:05d}.i3.bz2'
                # pathToInput = '/fhgfs/users/mhuennefeld/simulation/data/nugen_00001/04_level2/perfect_hex_grid/'
                # inputFileBaseName = 'nugen_01_level2_perfect_hex_grid_'
                gcdfile = directory_prefix + 'data/nugen_00001/configuration/GeoCalibDetectorStatus_2012.56063_V1_perfect_hex_grid.i3.gz'

                # define train, test and validation dataset
                fileNumbers_val = range(0,10)
                fileNumbers_test = range(10,20)
                fileNumbers_train = range(20,151) + range(333,478) + range(666,790)

            else:
                for key in ['pathToOutput','input_file_template',
                            'gcdfile','file_number_range_val',
                            'file_number_range_test','file_number_range_train']:

                            if key not in settings.keys():
                                raise ValueError('Dataset {} is unknown. Key {} must be defined'.format(settings['dataset'], key))

                # todo clean this up, so this does not have to be done
                pathToOutput = settings['pathToOutput']
                del settings['pathToOutput']
                input_file_template = settings['input_file_template']
                gcdfile = settings['gcdfile']
                fileNumbers_val = range(*settings['file_number_range_val'])
                fileNumbers_test = range(*settings['file_number_range_test'])
                fileNumbers_train = range(*settings['file_number_range_train'])

            print 'Working on Phobos. Adjusting path to:',pathToOutput

            # Tensorflow Session Config
            try:
                import tensorflow as tf
                config = tf.ConfigProto(
                    gpu_options = tf.GPUOptions(
                                                #per_process_gpu_memory_fraction=0.5,
                                                allow_growth = True
                                                ),
                    device_count = {'GPU': 1},
                    # intra_op_parallelism_threads=1,
                    # inter_op_parallelism_threads=1,
                )
                # tf.device('/gpu:1')
            except ImportError as e:
                print e
                print 'Creating empty Tensorflow config.'
                config = None

            numOfThreads = 12
        else:

            #------------------------------------
            #           Madison
            #------------------------------------

            # Get directories for dataset
            if settings['dataset'] == '11069' and not 'pathToOutput' in settings:
                pathToOutput = '/data/user/mhuennefeld/11069/output/'
                # pathToInput = '/data/sim/IceCube/2012/filtered/level2/neutrino-generator/11069/'
                # inputFileBaseName = 'Level2_nugen_numu_IC86.2012.011069.0'
                input_file_template = '/data/sim/IceCube/2012/filtered/level2/neutrino-generator/11069/{}/Level2_nugen_numu_IC86.2012.011069.{:06d}.i3.bz2'
                gcdfile = '/data/sim/IceCube/2012/filtered/level2/neutrino-generator/11069/00000-00999/GeoCalibDetectorStatus_2012.56063_V1.i3.gz'

                # define train, test and validation dataset
                fileNumbers_val = range(0,100)
                fileNumbers_test = range(100,200)
                fileNumbers_train = range(200,6000)

            elif settings['dataset'] == '11069_2017OnlineL2' and not 'pathToOutput' in settings:
                pathToOutput = '/data/user/mhuennefeld/11069/2017OnlineL2/output/'
                # pathToInput = '/data/user/mhuennefeld/11069/2017OnlineL2/'
                # inputFileBaseName = '2017OnlineL2_nugen_numu_IC86.2012.011069.0'
                input_file_template = '/data/user/mhuennefeld/11069/2017OnlineL2/{}/2017OnlineL2_nugen_numu_IC86.2012.011069.{:06d}.i3.bz2'
                gcdfile = '/data/sim/IceCube/2012/filtered/level2/neutrino-generator/11069/00000-00999/GeoCalibDetectorStatus_2012.56063_V1.i3.gz'

                # define train, test and validation dataset
                fileNumbers_val = range(0,100)
                fileNumbers_test = range(100,200)
                fileNumbers_train = range(200,6000)


            elif settings['dataset'] == '11069_PS_final' and not 'pathToOutput' in settings:
                pathToOutput = '/data/user/mhuennefeld/11069_PS_final/output/'
                # pathToInput = '/data/ana/PointSource/IC86_2012_PS/files/sim/2012/neutrino-generator/11069/'
                # inputFileBaseName = 'Final_v2_nugen_numu_IC86.2012.011069.0'
                input_file_template = '/data/user/mhuennefeld/11069_PS_final/output/{}/Final_v2_nugen_numu_IC86.2012.011069.{:06d}.i3.bz2'
                gcdfile = '/data/sim/IceCube/2012/filtered/level2/neutrino-generator/11069/00000-00999/GeoCalibDetectorStatus_2012.56063_V1.i3.gz'

                # define train, test and validation dataset
                fileNumbers_val = range(0,100)
                fileNumbers_test = range(100,200)
                fileNumbers_train = range(200,6000)

            else:
                for key in ['pathToOutput','input_file_template',
                            'gcdfile','file_number_range_val',
                            'file_number_range_test','file_number_range_train']:

                            if key not in settings.keys():
                                raise ValueError('Dataset {} is unknown. Key {} must be defined'.format(settings['dataset'], key))

                # todo clean this up, so this does not have to be done
                pathToOutput = settings['pathToOutput']
                del settings['pathToOutput']
                input_file_template = settings['input_file_template']
                gcdfile = settings['gcdfile']
                fileNumbers_val = range(*settings['file_number_range_val'])
                fileNumbers_test = range(*settings['file_number_range_test'])
                fileNumbers_train = range(*settings['file_number_range_train'])

            print 'Working on Madison Cluster. Adjusting path to:',pathToOutput

            # Tensorflow Session Config
            try:
                import tensorflow as tf
                config = tf.ConfigProto()
            except ImportError as e:
                print e
                print 'Creating empty Tensorflow config.'
                config = None
            numOfThreads = 4

        pathToOutput += outputFileAddition

        if 'cascade' == settings['dataset'][:7] or \
               settings['dataset'] in ['11069'] or \
               'Level2_muon_resimulation' in settings['dataset']:
            filelist_val = [pathToOutput+'{0:02d}000-{0:02d}999/{2}DOMPulseData_{1:08d}.hdf5'.format(int(n/1000),n,settings['dataset'])
                                for n in fileNumbers_val]

            filelist_test = [pathToOutput+'{0:02d}000-{0:02d}999/{2}DOMPulseData_{1:08d}.hdf5'.format(int(n/1000),n,settings['dataset'])
                                for n in fileNumbers_test]*1000

            filelist_train = [pathToOutput+'{0:02d}000-{0:02d}999/{2}DOMPulseData_{1:08d}.hdf5'.format(int(n/1000),n,settings['dataset'])
                                for n in fileNumbers_train]*100 # 100 epochs
        else:
            filelist_val = [pathToOutput+'{0:02d}000-{0:02d}999/{2}DOMPulseData_{1:04d}.hdf5'.format(int(n/1000),n,settings['dataset'])
                            for n in fileNumbers_val]

            filelist_test = [pathToOutput+'{0:02d}000-{0:02d}999/{2}DOMPulseData_{1:04d}.hdf5'.format(int(n/1000),n,settings['dataset'])
                            for n in fileNumbers_test]*1000

            filelist_train = [pathToOutput+'{0:02d}000-{0:02d}999/{2}DOMPulseData_{1:04d}.hdf5'.format(int(n/1000),n,settings['dataset'])
                            for n in fileNumbers_train]*100 # 100 epochs

        #------------------------------
        # Include additional datasets
        #------------------------------
        if 'add_datasets' in settings:
            assert len(settings['add_datasets']) == len(settings['add_pathToOutput'])
            assert len(settings['add_datasets']) == len(settings['add_file_number_range_val'])
            assert len(settings['add_datasets']) == len(settings['add_file_number_range_test'])
            assert len(settings['add_datasets']) == len(settings['add_file_number_range_train'])

            if 'dataset_weights' in settings:
                assert len(settings['add_datasets']) + 1 == len(settings['dataset_weights'])
                filelist_test = filelist_test * settings['dataset_weights'][0]
                filelist_train = filelist_train * settings['dataset_weights'][0]

            for d, dir_out, r_val, r_test, r_train, w in zip( settings['add_datasets'],
                                                           settings['add_pathToOutput'],
                                                           settings['add_file_number_range_val'],
                                                           settings['add_file_number_range_test'],
                                                           settings['add_file_number_range_train'],
                                                           settings['dataset_weights'][1:]
                                                            ):
                if 'cascade' == settings['dataset'][:7] or \
                       settings['dataset'] in ['11069'] or \
                       'Level2_muon_resimulation' in settings['dataset']:
                    files_val = [dir_out + outputFileAddition +'{0:02d}000-{0:02d}999/{2}DOMPulseData_{1:08d}.hdf5'.format( int(n/1000), n, d)
                                for n in range(*r_val)]

                    files_test = [dir_out + outputFileAddition +'{0:02d}000-{0:02d}999/{2}DOMPulseData_{1:08d}.hdf5'.format( int(n/1000), n, d)
                                for n in range(*r_test)] * 200 * w

                    files_train = [dir_out + outputFileAddition +'{0:02d}000-{0:02d}999/{2}DOMPulseData_{1:08d}.hdf5'.format( int(n/1000), n, d)
                                for n in range(*r_train)] * 200 * w
                else:
                    files_val = [dir_out + outputFileAddition +'{0:02d}000-{0:02d}999/{2}DOMPulseData_{1:04d}.hdf5'.format( int(n/1000), n, d)
                                for n in range(*r_val)]

                    files_test = [dir_out + outputFileAddition +'{0:02d}000-{0:02d}999/{2}DOMPulseData_{1:04d}.hdf5'.format( int(n/1000), n, d)
                                for n in range(*r_test)] * 200 * w

                    files_train = [dir_out + outputFileAddition +'{0:02d}000-{0:02d}999/{2}DOMPulseData_{1:04d}.hdf5'.format( int(n/1000), n, d)
                                for n in range(*r_train)] * 200 * w

                filelist_val.extend(files_val)
                filelist_test.extend(files_test)
                filelist_train.extend(files_train)
        #------------------------------

        train_batch_size = settings['remote_train_batch_size']
        test_batch_size = settings['remote_test_batch_size']

        # File list to create normModel
        if 'num_files_for_norm_model' in settings:
            num_files_for_norm_model = settings['num_files_for_norm_model']
        else:
            num_files_for_norm_model = 300
        filelist_norm_train = np.random.choice(filelist_train, num_files_for_norm_model, replace=False)
        filelist_norm_val = np.random.choice(filelist_train, 50, replace=False)
        # fileNumbers = range(200,500) #(200,500)
        # filelist_norm_train = [pathToOutput+'{0:02d}000-{0:02d}999/{2}DOMPulseData_{1:04d}.hdf5'.format(int(n/1000),n,settings['dataset']) for n in fileNumbers]

        # fileNumbers = range(900,950) # (900,950)
        # filelist_norm_val = [pathToOutput+'{0:02d}000-{0:02d}999/{2}DOMPulseData_{1:04d}.hdf5'.format(int(n/1000),n,settings['dataset']) for n in fileNumbers]


        # # # Test with single 700 event files
        # file_test = pathToOutput+'00000-00999/11069DOMPulseData_0000.hdf5'
        file_test = filelist_test[0]
        # # file_val = '/data/user/mhuennefeld/11069/output/00000-00999/11069PulseData_0001.hdf5'
        # # filelist_train = np.tile(['/data/user/mhuennefeld/11069/output/00000-00999/11069PulseData_{:04d}.hdf5'.format(n) for n in fileNumbers],10) # with 10*10 = 100 epochs

        # splines
        BareMuTimingSpline = '/cvmfs/icecube.opensciencegrid.org/data/photon-tables/splines/InfBareMu_mie_prob_z20a10_V2.fits'
        BareMuAmplitudeSpline = '/cvmfs/icecube.opensciencegrid.org/data/photon-tables/splines/InfBareMu_mie_abs_z20a10_V2.fits'
    # ---------------------------------------------------------------------

    if settings['labelsToLoad'] in ['labels123','labels126','labels131','labels134','labels137', 'labels143']:
        primaryEnergyKey = 'PrimaryEnergy'
        PrimaryAzimuthKey = 'PrimaryAzimuth'
        PrimaryZenithKey = 'PrimaryZenith'
        PrimaryMuonAzimuthKey = 'PrimaryMuonAzimuth'
        PrimaryMuonZenithKey = 'PrimaryMuonZenith'
        PrimaryMuonEnergyCenterKey = 'PrimaryMuonEnergyCenter'
        num_labels = int(settings['labelsToLoad'][-3:])
        # num_labels is not allowed to be 29, or 11!!!!!!!
        # otherwise old models for labels29 and labels11 will be overwritten
    elif settings['labelsToLoad'] == 'labels32':
        primaryEnergyKey = 'MCPrimary'
        PrimaryAzimuthKey = 'MCPrimaryAzimuth'
        PrimaryZenithKey = 'MCPrimaryZenith'
        PrimaryMuonAzimuthKey = 'MCPrimaryMuonAzimuth'
        PrimaryMuonZenithKey = 'MCPrimaryMuonZenith'
        PrimaryMuonEnergyCenterKey = 'MCPrimaryMuonEnergyCenter'
        num_labels = 32
    elif settings['labelsToLoad'] == 'labels29':
        primaryEnergyKey = 'MCPrimary'
        PrimaryAzimuthKey = 'MCPrimaryAzimuth'
        PrimaryZenithKey = 'MCPrimaryZenith'
        PrimaryMuonAzimuthKey = 'MCPrimaryMuonAzimuth'
        PrimaryMuonZenithKey = 'MCPrimaryMuonZenith'
        PrimaryMuonEnergyCenterKey = 'MCPrimaryMuonEnergyCenter'
        num_labels = 29
    elif settings['labelsToLoad'] == 'labels11':
        primaryEnergyKey = 'MCPrimary'
        PrimaryAzimuthKey = 'MCPrimaryAzimuth'
        PrimaryZenithKey = 'MCPrimaryZenith'
        PrimaryMuonAzimuthKey = 'MCPrimaryMuonAzimuth'
        PrimaryMuonZenithKey = 'MCPrimaryMuonZenith'
        PrimaryMuonEnergyCenterKey = 'MCPrimaryMuonEnergyCenter'
        num_labels = 11
    elif settings['labelsToLoad'] == 'CherenkovPositions':
        primaryEnergyKey = 'PrimaryEnergy'
        PrimaryAzimuthKey = 'PrimaryAzimuth'
        PrimaryZenithKey = 'PrimaryZenith'
        PrimaryMuonAzimuthKey = 'PrimaryMuonAzimuth'
        PrimaryMuonZenithKey = 'PrimaryMuonZenith'
        PrimaryMuonEnergyCenterKey = 'PrimaryMuonEnergyCenter'
        num_labels = 135 + (10*10*60 + 8*60)*4 # 25920 labels
    elif settings['labelsToLoad'] in ['CherenkovInfo','CherenkovInfo2']:
        primaryEnergyKey = 'PrimaryEnergy'
        PrimaryAzimuthKey = 'PrimaryAzimuth'
        PrimaryZenithKey = 'PrimaryZenith'
        PrimaryMuonAzimuthKey = 'PrimaryMuonAzimuth'
        PrimaryMuonZenithKey = 'PrimaryMuonZenith'
        PrimaryMuonEnergyCenterKey = 'PrimaryMuonEnergyCenter'
        num_labels = 135 + (10*10*60 + 8*60)*5
    elif settings['labelsToLoad'] == 'NuE_labels':
        primaryEnergyKey = 'PrimaryEnergy'
        PrimaryAzimuthKey = 'PrimaryAzimuth'
        PrimaryZenithKey = 'PrimaryZenith'
        PrimaryMuonAzimuthKey = None
        PrimaryMuonZenithKey = None
        PrimaryMuonEnergyCenterKey = None
        num_labels = 6
    elif settings['labelsToLoad'] == 'cascade_labels':
        primaryEnergyKey = 'PrimaryEnergy'
        PrimaryAzimuthKey = 'PrimaryAzimuth'
        PrimaryZenithKey = 'PrimaryZenith'
        PrimaryMuonAzimuthKey = None
        PrimaryMuonZenithKey = None
        PrimaryMuonEnergyCenterKey = None
        num_labels = 10
    elif settings['labelsToLoad'] == 'cascade_labels_pid':
        primaryEnergyKey = 'PrimaryEnergy'
        PrimaryAzimuthKey = 'PrimaryAzimuth'
        PrimaryZenithKey = 'PrimaryZenith'
        PrimaryMuonAzimuthKey = None
        PrimaryMuonZenithKey = None
        PrimaryMuonEnergyCenterKey = None
        num_labels = 25
    elif settings['labelsToLoad'] == 'cascade_vertex_labels':
        primaryEnergyKey = 'PrimaryEnergy'
        PrimaryAzimuthKey = 'PrimaryAzimuth'
        PrimaryZenithKey = 'PrimaryZenith'
        PrimaryMuonAzimuthKey = None
        PrimaryMuonZenithKey = None
        PrimaryMuonEnergyCenterKey = None
        num_labels = 10
    elif settings['labelsToLoad'][:18] == 'muon_energy_losses':
        primaryEnergyKey = None
        PrimaryAzimuthKey = None
        PrimaryZenithKey = None
        PrimaryMuonAzimuthKey = None
        PrimaryMuonZenithKey = None
        PrimaryMuonEnergyCenterKey = None

        if 'merge' in settings['labelsToLoad']:
              # get number of bins to merge from labels name
              # expects _merge{n_bins} to be in label name
              merge_str = [s for s in settings['labelsToLoad'].split('_')
                                          if 'merge' in s][0]
              n_bins_to_merge = int(merge_str[5:])

              # merge bins and labels
              n_bins = int(settings['labelsToLoad'][19:23])
              split_indices = np.arange(n_bins_to_merge, n_bins, n_bins_to_merge)
              num_labels = len( np.array_split( range(n_bins), split_indices))
        else:
            num_labels = int(settings['labelsToLoad'][19:23])

    else:
        raise ValueError('Unknown labels: {}'.format(settings['labelsToLoad']))

    normModelFile = settings['model_path']+'normModel_{}_{}_L{:02d}S{:02d}B{:02d}C{:01d}.npy'.format(settings['dataset'],settings['addWhichData'],num_labels,numberOfSummaryVars,numberOfBins,settings['addDOMCoordinates'])
    # ---------------------------------------------------------------------
    if settings['addDOMCoordinates'] or settings['labelsToLoad'] in \
        ['CherenkovPositions','CherenkovInfo','CherenkovInfo2', 'labels137', 'labels143']:
        # # check if domPosDictFile exists, if not create one
        # if not os.path.isfile(pathToOutput+'domPosDictFile.pkl'):

            # print(pathToOutput+'domPosDictFile.pkl','does not exist.')
            # print('Now creating one with gcd file:',gcdfile)

        if True: # force to make new file
            if not os.path.isdir(pathToOutput):
                os.makedirs(pathToOutput)
                print('\033[93mCreating directory: {}\033[0m'.format(pathToOutput))

            # get dictionary with xyz positions of each dom
            from icecube import dataio, dataclasses
            f = dataio.I3File(os.path.expanduser(gcdfile))
            f.pop_frame()
            i3geometry = f.pop_frame()['I3Geometry']
            geoMap = i3geometry.omgeo
            domPosDict = {(i[0][0],i[0][1]):(i[1].position.x,i[1].position.y,i[1].position.z) for i in geoMap if i[1].omtype.name == 'IceCube'}

            with open(pathToOutput+'domPosDictFile.pkl','w') as domPosDict_file:
                pickle.dump(domPosDict, domPosDict_file, pickle.HIGHEST_PROTOCOL)
            f.close()

        with open(pathToOutput+'domPosDictFile.pkl', 'rb') as f:
            domPulseDict =  pickle.load(f)
    else:
        domPulseDict = ''

    new_settings = {
        'primaryEnergyKey' : primaryEnergyKey,
        'PrimaryAzimuthKey' : PrimaryAzimuthKey,
        'PrimaryZenithKey' : PrimaryZenithKey,
        'PrimaryMuonAzimuthKey' : PrimaryMuonAzimuthKey,
        'PrimaryMuonZenithKey' : PrimaryMuonZenithKey,
        'PrimaryMuonEnergyCenterKey' : PrimaryMuonEnergyCenterKey,
        'num_labels' : num_labels,
        'numberOfBins' : numberOfBins,
        'numberOfSummaryVars' : numberOfSummaryVars,
        'normModelFile' : normModelFile,
        'domPulseDict' : domPulseDict,
        'config' : config,
        'file_test' : file_test,
        'filelist_train' : filelist_train,
        'filelist_test' : filelist_test,
        'filelist_val' : filelist_val,
        'filelist_norm_val' : filelist_norm_val,
        'filelist_norm_train' : filelist_norm_train,
        'filelist_val' : filelist_val,
        'numOfThreads' : numOfThreads,
        'train_batch_size': train_batch_size,
        'test_batch_size': test_batch_size,
        'outputFileAddition': outputFileAddition,
        'gcdfile': gcdfile,
        'pathToOutput': pathToOutput,
        # 'pathToInput' : pathToInput,
        # 'inputFileBaseName' : inputFileBaseName,
        'input_file_template' : input_file_template,
        'BareMuAmplitudeSpline' : BareMuAmplitudeSpline,
        'BareMuTimingSpline' : BareMuTimingSpline,
        }
    # settings.update(new_settings) #  overwrite keys in dictionary settings
    settings = dict( new_settings.items() + settings.items() ) # don't overwrite keys in dictionary settings
    return settings

