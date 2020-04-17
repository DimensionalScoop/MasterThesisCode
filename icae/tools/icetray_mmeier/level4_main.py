# /usr/bin/env python
# -*-coding:utf-8-*-

from __future__ import division, print_function

import numpy
import sys
import os

from I3Tray import *
from icecube import icetray, dataclasses, dataio

# import Filter
# import base_process
# import Waveform_Calibration
# import calc_attributes
# import Qtot
# from pulse_cleaning import PulseNamingAndCleaning
# from reconstructions.cscd_l3_recos import Cascade_L3_Recos
# from modules.feature_generation import feature_generation_segment


if __file__.endswith("c"):
    SCRIPT_NAME = __file__[:-1]
else:
    SCRIPT_NAME = __file__

CMD = "python {} -i $(inputfile) -o $(outputfile) -g $(gcdfile) -d $(dataset) -w $(generator)".format(
    SCRIPT_NAME
)
PROCESS_NAME = "level4"


def main(in_files, out_file, gcd_file, dataset, generator=None):
    import Filter
    import Waveform_Calibration
    import calc_attributes
    import Qtot
    from pulse_cleaning import PulseNamingAndCleaning
    from reconstructions.cscd_l3_recos import Cascade_L3_Recos
    from modules.feature_generation import feature_generation_segment

    files = []
    if isinstance(gcd_file, list) and isinstance(in_files, list):
        if len(gcd_file) == len(in_files):
            from itertools import chain

            files = list(chain(*zip(gcd_file, in_files)))
        else:
            files.extend(gcd_file)
            files.extend(in_files)
    else:
        raise TypeError("Both 'gcd_file' and 'in_files' are " + "expected to be lists.")

    if dataset is not None:
        datatype = dataset["type"].lower()
    else:
        datatype = "data"

    tray = I3Tray()
    # Read files
    tray.AddModule("I3Reader", "reader", Filenamelist=files)

    # Run ID Correction for bugged simulation sets
    if dataset is not None:
        if dataset["type"] == "corsika":
            if int(dataset["year"]) == 2012 or dataset["datasetnumber"] in [10688]:
                print("Running Run ID Correction")
                from run_id_correction import run_id_corrector

                tray.AddModule(run_id_corrector, "Correct_Run_IDs", i3_files=in_files)
            if dataset["datasetnumber"] in [10784, 10651, 10281]:
                print("Running a special Run ID Correctin for broken 2011 sets")
                from run_id_correction import run_id_corrector_2011

                tray.AddModule(
                    run_id_corrector_2011,
                    "Correct_Run_IDs_2011_style",
                    i3_files=in_files,
                    dataset_folder=dataset["madison_path"],
                )
        if dataset["datasetnumber"] in [14012, 14014, 14016]:
            from run_id_correction import run_id_corrector_nancy

            tray.AddModule(
                run_id_corrector_nancy,
                "Correct RunIDs from Nancy",
                i3_files=in_files,
                dataset_number=dataset["datasetnumber"],
            )

    if datatype == "muongun":
        from modules.muongun_preprocessing import prepare_muongun

        if len(gcd_file) == 1:
            gcd_file = gcd_file[0]
        tray.AddSegment(
            prepare_muongun,
            "muongun_preproc",
            gcd_file=gcd_file,
            in_file=in_files[-1],
            dataset=dataset,
        )

    # Apply EHE Filter for the desired detector configuration
    if dataset["detector_type"] == "IC86-II":
        tray.AddSegment(Filter.Filter12, "EHEFilter")
    elif dataset["detector_type"] == "IC86-I":
        tray.AddSegment(Filter.Filter11, "EHEFilter")
    elif dataset["detector_type"] == "IC79":
        tray.AddSegment(Filter.Filter10, "EHEFilter")
    elif dataset["detector_type"] == "IC86-III":
        tray.AddModule(Filter.Filter13, "EHEFilter")

    # Pulse renaming and HLC Cleaning
    tray.AddSegment(PulseNamingAndCleaning, "NameAndCleanPulses", dataset=dataset)

    # Calc total charge
    tray.AddSegment(Qtot.CalcQTot, "Qtot", pulses="OfflinePulses")

    tray.AddModule(lambda frame: numpy.log10(frame["CausalQTot"].value) >= 3.3)

    # Calibration Waveforms from DomLaunches and do SRTCleaning
    tray.AddSegment(Waveform_Calibration.CalibrationAndCleaning, "CalibAndCling")

    # Calc attributes to characterize the waveform and apply a random forest
    tray.AddModule(calc_attributes.calc_attributes, "cattr")

    # Delete huge objects in the frame
    to_delete = [
        "CalibratedWaveforms",
        "CalibratedWaveformsSLC",
        "CalibratedWaveformsHLCFADC",
        "CalibratedWaveformsHLCATWD",
        "I3MCPESeriesMap",
        "I3MCPulseSeriesMap",
        "I3MCPulseSeriesMapParticleIDMap",
    ]
    tray.AddModule("Delete", "delete", Keys=to_delete)

    if datatype != "data":
        if datatype in ["nutau", "numu", "nue"]:
            from resources import fluxes_neutrino as flx_gen
        elif datatype == "corsika":
            from resources import fluxes_corsika as flx_gen
        if datatype == "muongun":
            from resources import fluxes_muongun as flx_gen

            generator = flx_gen.harvest_generators(
                in_files, n_files=dataset["n_files_madison"], equal_generators=True
            )

        fluxes, flux_names = flx_gen.get_fluxes_and_names()
        import weights

        tray.AddSegment(
            weights.do_the_weighting,
            "Weighting",
            fluxes=fluxes,
            flux_names=flux_names,
            dataset=dataset,
            generator=generator,
            key="Weights",
        )

    if datatype == "nutau":
        # Check the actual neutrino flavor interacting in the detector
        import in_ice_labels

        tray.AddModule(in_ice_labels.in_ice_labels, "in_ice_labels")

        # Check the containment of both cascade vertices
        from modules.containment_label import containment_label

        tray.AddModule(containment_label, "ContLabel")

        from modules.tau_infos import tau_infos

        tray.AddModule(tau_infos, "TauInfos")

    if datatype == "corsika":
        from modules.multiplicity_module import BundleCharacterization

        tray.AddModule(BundleCharacterization, "Char bundles", fast=True)

    if datatype == "muongun":
        from modules.muongun_labels import muongun_labels

        tray.AddModule(muongun_labels, "Label module")

    # Cascade L3 reconstructions without the application of L3 Cuts
    AMP_TABLE = (
        "/cvmfs/icecube.opensciencegrid.org/data/photon-tables/"
        + "splines/ems_mie_z20_a10.abs.fits"
    )
    TIME_TABLE = (
        "/cvmfs/icecube.opensciencegrid.org/data/photon-tables/"
        + "splines/ems_mie_z20_a10.prob.fits"
    )
    tray.AddSegment(
        Cascade_L3_Recos,
        "CascadeL3Recos",
        YEAR=dataset["year"],
        AMP_TABLE=AMP_TABLE,
        TIME_TABLE=TIME_TABLE,
    )

    # Do some more feature extraction:
    tray.AddSegment(feature_generation_segment, "feature_generation", gcd_file=gcd_file)

    # Write everything to i3 files
    tray.AddModule(
        "I3Writer",
        "writer",
        filename=out_file,
        Streams=[icetray.I3Frame.DAQ, icetray.I3Frame.Physics],
        DropOrphanStreams=[icetray.I3Frame.DAQ],
    )
    # Dropping Orphan Streams might result in empty files, which causes crashes

    tray.AddModule("TrashCan", "YesWeCan")
    tray.Execute()
    tray.Finish()


#################################################
# Test
#################################################
if __name__ == "__main__":
    from optparse import OptionParser
    from helper import assist_functions as af
    import os

    parser = OptionParser()
    parser.add_option("-i", "--infiles", type="string", dest="in_files")
    parser.add_option("-d", "--dataset", type="string", dest="dataset")
    parser.add_option("-g", "--gcdfile", type="string", dest="gcd_file")
    parser.add_option("-o", "--outputdir", type="string", dest="out_file")
    parser.add_option("-w", "--generator", type="string", dest="generator")
    (options, args) = parser.parse_args()

    in_files = options.in_files
    in_files = in_files.split(",")
    gcd_file = options.gcd_file
    gcd_file = gcd_file.split(",")
    out_file = options.out_file
    generator = options.generator
    if generator == "None":
        generator = None
    af.ensure_dir(os.path.join(os.path.dirname(out_file), ""))

    try:
        dataset = af.load_dataset(options.dataset)
    except IOError:
        dataset = af.load_detector_cart(options.dataset)
    main(in_files, out_file, gcd_file, dataset, generator)
