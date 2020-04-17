#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Icetray module calculating attributes based on the dom hits in an
event. A dom hit is defined by a integrated charge, time and position
(BEWARE: I call it DOM HIT, might not be the kind of DOM HIT everyone
else is talking about).
This module creates a dom_dict. The dom_dict discribes the detector in a
python dict containing reduce informations. For each event the module
extracts the hits with its properties.
This is done, do have all the informations a bit more handy
representation.
The extracted information is provides to three different classes:
- Dustyness
- Borderness
- ProjectedQ

Dustyness:
    This class calculates:
        - "Q_ratio_in_dust": Ratio of the charge inside the dust layer
        - "Q_ratio_out_dust": Ratio of the charge outside the dust layer
        - "n_doms_out_dust": Ratio of dom hits inside the dust layer
        - "n_doms_in_dust": Ratio of dom hits outside the dust layer

Borderness:
    This class calculates:
        - "Q_ratio_in_border": Ratio of the charge inside the boarder
        - "Q_ratio_in_center": Ratio of the charge outside the boarder
        - "n_doms_out_border": Ratio of dom hits inside the boarder
        - "n_doms_in_center": Ratio of dom hits outside the boarder

ProjectedQ:
    The idea of the calculations is to look right into the direction of
    the reconstructed track. and project all dom hits onto a plane,
    horizontal to the track direction. On this projections parameters
    similar to the "Hillas Parameters" (famous from cherenkov astronomy)
    are calculated. The idea of the Hillas Parameters is to describe the

    This class calculates:
        - "length": semi-major axis of the ellipse
        - "width": semi-minor axis of the ellipse
        - "area": area of the ellipse
        - "ratio": length/width
"""
from __future__ import division, print_function

import copy
import cPickle

import numpy as np

from icecube import dataclasses, icetray, dataio, common_variables

from reconstructions.pulse_calc import projected_charge, dustyness, borderness
from helper.assist_functions import add_dict_to_frame

border_strings = [
    1,
    2,
    3,
    4,
    5,
    6,
    13,
    21,
    30,
    40,
    50,
    59,
    67,
    74,
    73,
    72,
    78,
    77,
    76,
    75,
    68,
    60,
    51,
    41,
    31,
    22,
    14,
    7,
]
deepcore_string = [79, 80, 81, 82, 83, 84, 85, 86]


class calc_charge_attributes(icetray.I3ConditionalModule):
    """This module calculates Dustyness, Borderness and ProjectedQ
    attributes."""

    def __init__(self, context):
        """Pretty standard Configure() to set all needed parameters."""
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddParameter("gcd_path", "gcd_path", "")
        self.AddParameter("spline_mpe_key", "spline_mpe_key", "SplineMPE")
        self.AddParameter("line_fit_key", "line_fit_key", "LineFit_TT")
        self.AddParameter("hit_stats_key", "hit_stats_key", "HitStatisticsValues")
        self.AddParameter("pulse_map", "pulse_map", "SRTHVInIcePulses")

    def Configure(self):
        """Pretty standard Configure() to get all the set parameters."""
        self.gcd_path = self.GetParameter("gcd_path")
        self.spline_mpe_key = self.GetParameter("spline_mpe_key")
        self.line_fit_key = self.GetParameter("line_fit_key")
        self.hit_stats_key = self.GetParameter("hit_stats_key")
        self.pulse_map = self.GetParameter("pulse_map")

    class dom_dict_entry:
        """Class used to descripe a Dom in the Dom dict. Position and
        if the DOM is a border Dom is stored."""

        def __init__(self, position, border=False):
            self.pos = position
            self.border = border

    def get_top_bottom(self, dom_dict):
        """Function to determine if a Dom is in the Top or in the
        bottom layer of strings."""
        strings = {}
        om_key = {}
        for key in dom_dict:
            z = dom_dict[key].pos[2]
            string = key.string
            if string in deepcore_string:
                continue
            if (np.absolute(z) < 600.0) and (np.absolute(z) > 450.0):
                try:
                    min_max = strings[string]
                    if z > min_max[1]:
                        strings[string][1] = z
                        om_key[string][1] = key
                    if z < min_max[0]:
                        strings[string][0] = z
                        om_key[string][0] = key
                except:
                    strings[string] = [z, z]
                    om_key[string] = [key, key]
        return om_key

    def create_dom_dict(self, geometry_frame):
        """Create the dom_dict from a geometry frame."""
        dom_dict = {}
        i3geometry = geometry_frame["I3Geometry"].omgeo
        for key in i3geometry.keys():
            dom_pos = i3geometry[key].position
            if dom_pos.z > 600.0:
                border = False
            else:
                if key.string in border_strings:
                    border = True
                else:
                    border = False
            position = np.array([dom_pos.x, dom_pos.y, dom_pos.z])
            dom_dict[key] = self.dom_dict_entry(position, border)
        om_dict = self.get_top_bottom(dom_dict)
        for key in om_dict.keys():
            om_keys = om_dict[key]
            dom_dict[om_keys[0]].border = True
            dom_dict[om_keys[1]].border = True
        return dom_dict

    def get_dom_dict(self, frame, buffer_dict=False):
        """Functionality to save and reuse a dom dict."""
        if buffer_dict:
            dom_dict_path = self.gcd_path + ".dom_dict"
            try:
                with open(dom_dict_path, "r") as fp:
                    dom_dict = cPickle.load(fp)
            except:
                dom_dict = copy.deepcopy(self.create_dom_dict(frame))
                with open(dom_dict_path, "wb") as fp:
                    cPickle.dump(dom_dict, fp)
        else:
            dom_dict = self.create_dom_dict(frame)
        return dom_dict

    def Geometry(self, frame):
        """Create of fetch the dom_dict."""
        self.dom_dict = self.get_dom_dict(frame)
        self.PushFrame(frame)

    def Physics(self, frame):
        """Funciton handling P-Frames. Extracts the dom hits and
        provides them to the Borderness, Dustyness and ProjectedQ
        classes."""
        if isinstance(frame[self.pulse_map], dataclasses.I3RecoPulseSeriesMapMask):
            pulses = frame[self.pulse_map].apply(frame)
        elif isinstance(frame[self.pulse_map], dataclasses.I3RecoPulseSeriesMap):
            pulses = frame[self.pulse_map]
        else:
            raise TypeError(
                "Pulse Map has to be either a RecoPulseSeriesMap " "or a RPSMMask"
            )
        n_pulses = len(pulses)
        if n_pulses > 0:
            track = frame[self.spline_mpe_key].dir
            pos = frame[self.spline_mpe_key].pos
            cog = frame[self.hit_stats_key].cog
            cog = np.array([cog.x, cog.y, cog.z])
            zenith = track.zenith
            azimuth = track.azimuth
            origin = np.array([pos.x, pos.y, pos.z])
            proj_charge = projected_charge(zenith, azimuth, n_pulses, cog, origin)
            q_dust = dustyness()
            q_border = borderness()
            for i, dom_pulse in enumerate(pulses):
                charge = 0.0
                for p in dom_pulse[1]:
                    charge += p.charge
                proj_charge.add_point(charge, self.dom_dict[dom_pulse[0]])
                q_dust.add_point(charge, self.dom_dict[dom_pulse[0]])
                q_border.add_point(charge, self.dom_dict[dom_pulse[0]])
            add_dict_to_frame(frame, q_dust.return_value_dict(), "Dustyness")
            add_dict_to_frame(frame, q_border.return_value_dict(), "Borderness")
            add_dict_to_frame(frame, proj_charge.return_value_dict(), "ProjectedQ")
            self.PushFrame(frame)
