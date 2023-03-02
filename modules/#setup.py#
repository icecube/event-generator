from __future__ import print_function, division
import os
import importlib

import yaml
import glob
from copy import deepcopy

from . import file_utils
from .exp_data import livetime
from .exp_data.good_run_list_utils import get_gcd_from_grl


def setup_job_and_config(cfg, run_number, scratch, verbose=True):
    """Setup Config and Job settings

    Parameters
    ----------
    cfg : str or dict
        File path to config file.
    run_number : int
        The runnumber.
    scratch : bool
        Whether or not to run on scratch.
    verbose : bool, optional
        If True, print additional information.

    Returns
    -------
    dict
        The dictionary with settings.
    dict
        Additional output values.

    Raises
    ------
    IOError
        Description
    """

    additional_values = {}

    if isinstance(cfg, str):
        with open(cfg, 'r') as stream:
            cfg = yaml.full_load(stream)
    else:
        cfg = deepcopy(cfg)

    # update environment variables if provided
    if 'set_env_vars_from_python' in cfg:
        print('\n------------------------------------------------')
        print('Setting Environment Variables from within Python')
        print('------------------------------------------------')
        for key, value in cfg['set_env_vars_from_python'].items():
            os.environ[key] = str(value)
            print('    Setting {} to {}'.format(key, value))
        print('------------------------------------------------\n')

    cfg['run_number'] = run_number

    # Update config with run number specific settings if provided
    if 'run_number_settings' in cfg:
        cfg.update(cfg['run_number_settings'][run_number])

    cfg['folder_num_pre_offset'] = cfg['run_number']//1000
    cfg['folder_num'] = cfg['folder_offset'] + cfg['run_number']//1000
    cfg['folder_pattern'] = cfg['folder_pattern'].format(**cfg)
    cfg['run_folder'] = cfg['folder_pattern'].format(**cfg)

    # get GCD file for exp data run
    if cfg['gcd'] == 'GET_GCD_FROM_GRL':
        if verbose:
            print('Searching for GCD file for run {:08d}...'.format(
                run_number))

        cfg['gcd'] = get_gcd_from_grl(
            grl_patterns=cfg['exp_dataset_grl_paths'],
            run_id=run_number,
        )
        if verbose:
            print('\tFound: {}'.format(cfg['gcd']))

    if (cfg['merge_files'] or 'merge_input_files' in cfg or
            'merge_input_file_glob' in cfg):

        infiles = [cfg['gcd']]

        if 'merge_input_files' in cfg:
            merge_input_files = cfg['merge_input_files']
            no_file_err_msg = cfg['merge_input_files']

        elif 'merge_input_file_glob' in cfg:
            merge_input_files = sorted(glob.glob(
                cfg['merge_input_file_glob'].format(**cfg)))
            no_file_err_msg = cfg['merge_input_file_glob'].format(**cfg)

        else:
            # get all files in input run folder
            input_run_folder = os.path.dirname(
                cfg['in_file_pattern'].format(**cfg))
            merge_input_files = sorted(glob.glob(
                '{}/*.i3*'.format(input_run_folder)))
            no_file_err_msg = input_run_folder

        for name in merge_input_files:
            if not os.path.exists(name):
                raise IOError('Input file {} does not exist!'.format(name))
            infiles.append(name)

        # check if readable, or if file is corrupt
        if 'check_merged_files' in cfg and cfg['check_merged_files']:
            if verbose:
                print('Checking if files are readable...')
            filtered_infiles = []
            frame_cnt = 0
            for file_name in infiles:
                count_i = file_utils.file_is_readable(file_name)
                if count_i is None:
                    assert file_name != cfg['gcd'], 'GDC file is corrupt!'
                    if verbose:
                        print('Found possibly corrupt file: {}'.format(
                            file_name))
                else:
                    frame_cnt += count_i
                    filtered_infiles.append(file_name)
            if verbose:
                print('Filtered out {} files. Processing {} frames.'.format(
                    len(infiles) - len(filtered_infiles), frame_cnt))
            infiles = filtered_infiles

        # merge calculated weights and update n_files used in meta data
        if 'merge_weights' in cfg and cfg['merge_weights']:

            # get total number of files
            if verbose:
                print('Computing total of n_files...')
            total_n_files = file_utils.get_total_weight_n_files(infiles[1:])
            additional_values['total_n_files'] = total_n_files
            if verbose:
                print('Merging weights with a total of n_files = '
                      '{} over {} input files'.format(
                        total_n_files, len(infiles[1:])))

        # merge experimental livetime and update meta data in X-frame
        if 'exp_dataset_merge' in cfg and cfg['exp_dataset_merge']:

            # get livetime information from all files
            if verbose:
                print('Collecting exp livetime info...')
            cfg = livetime.collect_exp_livetime_data(infiles[1:], cfg)
            if verbose:
                msg = 'Collected livetime of {} days over {} input files'
                print(msg.format(
                    cfg['exp_dataset_livetime'] / 24. / 3600.,
                    len(infiles[1:])))

        n_files = len(infiles[1:])
        if n_files < 1:
            raise IOError('No files found for:\n\t {}'.format(no_file_err_msg))
        if verbose:
            print('Found {} files.'.format(n_files))
    else:
        # get input file path
        infile = cfg['in_file_pattern'].format(**cfg)

        if not os.path.exists(infile):
            raise IOError('Input file {} does not exist!'.format(infile))

        if not cfg['gcd'] is None:
            infiles = [cfg['gcd'], infile]
            n_files = len(infiles[1:])
        else:
            infiles = [infile]
            n_files = len(infiles)

    # add s-Frames if specified
    if 'sframes_to_load' in cfg and cfg['sframes_to_load']:
        if isinstance(cfg['sframes_to_load'], str):
            cfg['sframes_to_load'] = [cfg['sframes_to_load']]

        s_frame_files = []
        for s_frame in cfg['sframes_to_load']:
            s_frame_files.append(s_frame.format(**cfg))
        additional_values['s_frame_files'] = s_frame_files
        infiles = s_frame_files + infiles

    if scratch:
        outfile = cfg['scratchfile_pattern'].format(**cfg)
        scratch_output_folder = os.path.dirname(outfile)
        if scratch_output_folder and not os.path.isdir(scratch_output_folder):
            os.makedirs(scratch_output_folder)
    else:
        outfile = os.path.join(cfg['data_folder'],
                               cfg['out_dir_pattern'].format(**cfg))
        if not cfg['merge_files']:
                    outfile = os.path.join(outfile,
                                           cfg['run_folder'].format(**cfg))
        outfile = os.path.join(outfile, cfg['out_file_pattern'].format(**cfg))

    return cfg, infiles, n_files, outfile, additional_values


def load_class(full_class_string):
    """
    dynamically load a class from a string

    Parameters
    ----------
    full_class_string : str
        The full class string to the given python clas.
        Example:
            my_project.my_module.my_class

    Returns
    -------
    python class
        PYthon class defined by the 'full_class_string'
    """

    class_data = full_class_string.split(".")
    module_path = ".".join(class_data[:-1])
    class_str = class_data[-1]

    module = importlib.import_module(module_path)
    # Finally, we retrieve the Class
    return getattr(module, class_str)


def get_full_class_string_of_object(object_instance):
    """Get full class string of an object's class.

    o.__module__ + "." + o.__class__.__qualname__ is an example in
    this context of H.L. Mencken's "neat, plausible, and wrong."
    Python makes no guarantees as to whether the __module__ special
    attribute is defined, so we take a more circumspect approach.
    Alas, the module name is explicitly excluded from __qualname__
    in Python 3.

    Adopted from:
        https://stackoverflow.com/questions/2020014/
        get-fully-qualified-class-name-of-an-object-in-python

    Parameters
    ----------
    object_instance : object
        The object of which to obtain the full class string.

    Returns
    -------
    str
        The full class string of the object's class
    """
    module = object_instance.__class__.__module__
    if module is None or module == str.__class__.__module__:
        # Avoid reporting __builtin__
        return object_instance.__class__.__name__
    else:
        return module + '.' + object_instance.__class__.__name__
