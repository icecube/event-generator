"""
Helper-functions for file utilities
"""
from icecube import dataio, icetray


def file_is_readable(file_name):
    """Check if an I3 file is readable, or if it is corrupt.

    Parameters
    ----------
    file_name : str
        The path to the file.

    Returns
    -------
    int
        Number of frames. If None, then something is wrong with the file.
        For instance, it may be corrupt.
    """
    frame_counter = 0
    try:
        f = dataio.I3File(file_name)
        while f.more():
            fr = f.pop_frame()
            frame_counter += 1
        f.close()
    except Exception as e:
        print('Caught an excpetion:', e)
        frame_counter = None
    return frame_counter


def get_total_weight_n_files(file_names):
    """Get the total number of files merged together.

    This is the n_files parameter for weighting.

    Warning: this breaks down if a file does not have any frame with a
    weights key! Make sure

    Parameters
    ----------
    file_name : list of str
        A list of input files.

    Returns
    -------
    int
        Number of frames. If None, then something is wrong with the file.
        For instance, it may be corrupt.
    """

    total_n_files = 0

    for file_name in file_names:
        f = dataio.I3File(file_name)
        while f.more():
            fr = f.pop_frame()
            if 'weights_meta_info' in fr:
                total_n_files += fr['weights_meta_info']['n_files']
                break
            elif (fr.Stop == icetray.I3Frame.Stream('W') or
                  fr.Stop == icetray.I3Frame.Physics):
                raise ValueError('No weight meta data found!')
        f.close()

    return total_n_files
