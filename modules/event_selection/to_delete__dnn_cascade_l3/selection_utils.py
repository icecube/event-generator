

def discard_events_not_passing_l2_cascade_filter(frame):
    """Discard events that did not pass the Level2 Cascade Filter.

    Parameters
    ----------
    frame : I3Frame
        Current I3Frame.

    Returns
    -------
    bool
        True if cascade or muon filter is passed.

    Raises
    ------
    ValueError
        If keys can't be found.
    """
    if 'FilterMask' in frame:
        f = frame['FilterMask']
    elif 'QFilterMask' in frame:
        f = frame['QFilterMask']
    else:
        raise ValueError('No FilterMask in {!r}'.format(frame))

    cascade_key = None
    for key in f.keys():
        if 'CascadeFilter' == key[:13]:
            cascade_key = key

    if cascade_key is None:
        raise ValueError(
            'Could not find CascadeFilter in {!r}'.format(f.keys()))

    return f[cascade_key].condition_passed
