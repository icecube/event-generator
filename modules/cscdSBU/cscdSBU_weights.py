from icecube import icetray, dataclasses
from icecube.icetray import traysegment

from .nugen import nugen_weights, nugen_truth


@icetray.traysegment
def weights(tray, name, datatype='data'):

    if datatype not in ['nugen', 'muongun', 'data', 'corsika']:
        raise ValueError('Unknown datatype: {!r}'.format(datatype))

    if datatype == 'nugen':
        # add atmospheric neutrino weights
        from .nugen import nugen_weights
        tray.AddSegment(nugen_weights, 'atm_flux',
                        If=lambda f: 'cscdSBU_MCPrimary' not in f)

        from .nugen import nugen_truth
        tray.AddSegment(nugen_truth, 'visible_truth',
                        If=lambda f: 'cscdSBU_MCTruth' not in f)
