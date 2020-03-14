#! usr/bin/env python
# -*- coding: utf-8 -*-

# Python script for generating a spectrogramm with the template
# best fitting the data

# Author: David Corre
# email: corre@lal.in2p3.fr


"""
Packages required:
- gwpy
- python-lalsimulation
- pycbc
- matplotlib
- numpy

If used on Virgo server:
- PythonVirgoTools

If used outside Virgo server:
- python-nds2-client 

Requirements for templates:
Either installed the whole package, but about 20GB
- lalsuite-extra

Or simply download this file:
https://git.ligo.org/lscsoft/lalsuite-extra/blob/master/data/lalsimulation/SEOBNRv4ROM_v2.0.hdf5
and set an environment variable $LAL_DATA_PATH to the location of this file.

Waveform template: SEOBNRv4ROM_v2.0
https://ui.adsabs.harvard.edu/abs/2017PhRvD..95d4028B/abstract
--------------------------------
Example of usage on the Virgo server:

# Event S200129m / G361581
# Constrain time intervall to determine loudest Q-plane
python spectro_template.py --tstart 1264316106 --tend_or_dur 1264316127 --m1 47.73967 --m2 29.955526 --s1 0.83113605 --s2 -0.98355848 --gps_event 1264316116.4351 --useGPS True --search 0.2

--------------------------------

# With 2 inspiral template signals
python spectro_template.py --tstart 1264316106 --tend_or_dur 1264316127 --m1 47.73967 --m2 29.955526 --s1 0.83113605 --s2 -0.98355848 --gps_event 1264316116.4 1264316116.5 --verbose 

--------------------------------

Example of usage outside the Virgo server
(Can take some time to retrieve the data)
Set the --remote argument
You will be asked to provide your ligo.org username and password.

python spectro_template.py --tstart 1264316114 --tend_or_dur 1264316118 --m1 47.73967 --m2 29.955526 --s1 0.83113605 --s2 -0.98355848 --gps_event 1264316116.4351 --useGPS True --search 0.2 --twindow 1 --remote --whiten False 

# Using 3 time windows, 0.5s, 2s and 10s
python spectro_template.py --tstart 1264316106 --tend_or_dur 1264316127 --m1 47.73967 --m2 29.955526 --s1 0.83113605 --s2 -0.98355848 --gps_event 1264316116.4351 --useGPS True --search 0.2 --remote --whiten False --frange 20 1400


--------------------------------

Example of usage outside the Virgo server, using L1 data

# Using a single time window of 1s around the gps event
python spectro_template.py --channel L1:GDS-CALIB_STRAIN_CLEAN --tstart 1264316114 --tend_or_dur 1264316118 --m1 47.73967 --m2 29.955526 --s1 0.83113605 --s2 -0.98355848 --gps_event 1264316116.4351 --useGPS True --search 0.2 --twindow 1 --remote --frange 20 1400

# Using 3 time windows, 0.5s, 2s and 10s
python spectro_template.py --channel L1:GDS-CALIB_STRAIN_CLEAN --tstart 1264316106 --tend_or_dur 1264316127 --m1 47.73967 --m2 29.955526 --s1 0.83113605 --s2 -0.98355848 --gps_event 1264316116.4351 --useGPS True --search 0.2 --remote --frange 20 1400


--------------------------------

Typical errors

If you get the error `window is longer than input signal`, 
increase the time interval when querying the data, i.e.
`tstart` and / or `tend_or_dur`.

If you get the following error `FloatingPointError: divide
by zero encountered in true_divide` due to 0.0 values in
asd.value. Set `whiten` to `False`.

"""


import traceback
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from gwpy.time import to_gps
from gwpy.timeseries import TimeSeries
from gwpy.plot import Plot
from gwpy.signal import qtransform
from gwpy.segments import Segment
from pycbc.pnutils import get_inspiral_tf

def read_virgo_timeseries(source, channel, t0, gstop_or_dur,
                          mask=False, fill_value=np.nan,
                          remote=False):
    """Function to read virgo data as timeseries.
       This should one day be included in gwpy.
   
    Parameters
    ----------
    source : `str`
        Frame file, either a full path to a ffl or gwf file, or an
        abbreviation like 'raw', 'trend', which are looked up in
        the stancard location. If omitted, defaults to 'raw',
        but this default value is deprecated.

    channel : `str`
        Source datastream for these data. 
        If missing, a prefix 'V1:' is added.

    t0 : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
        GPS epoch corresponding to starting time,
        any input parsable by `~gwpy.time.to_gps` is fine

    gstop_or_dur: `~gwpy.time.LIGOTimeGPS`, `float`, `str`
        GPS epoch corresponding to end time,
        any input parsable by `~gwpy.time.to_gps` is fine
        If a `float` < 1e6 is provided, it corresponds to a duration
        in seconds from `t0`.

    mask : `bool`, optional
        If mask is False, missing samples will be replaced
        by fill_value. If it is True, the returned FrVect will
        have an attribute missing, which is a mask vector that
        is zero for missing samples.
        Default : False

    fill_value : `float`, optional
        Value that is used for missing samples if mask is False.
        Default: np.nan

    remote: `bool` optional
        If False, use PythonVirgoTools to parse raw data files.
        If True, use gwpy.TimeSeries.get(), but takes longer.
        Default : False, assuming the script is ran on Virgo server.

    Examples
    --------
    >>> from virgotools import getChannel
 
    Load the data from channel 'INJ_IMC_TRA_DC', from
    Sep 14 2015 09:50:45.391, and a duration of 10s

    >>> x = getChannel('raw', 'INJ_IMC_TRA_DC', 'Sep 14 2015 09:50:45.391', 10)

    That can be simply visualise:

    >>> import matplotlib.pyplot as plt
    >>> plt.plot(x.time, x.data)
    >>> plt.show()

    Same, using 2 GPS times:

    >>> x = getChannel('raw', 'INJ_IMC_TRA_DC', 1126259462.3910, 1126259472.3910)

    """

    # Convert to gps times in seconds
    # Use the Seconds and NanoSeconds instead of ns()
    # Because otherwise one needs to multiply by 1e-9
    # And this can cause rounding approximation 
    sec = to_gps(t0).gpsSeconds
    nsec = to_gps(t0).gpsNanoSeconds
    tstart = str(sec) + '.' + str(nsec)
    gstart = float(tstart)

    sec = to_gps(gstop_or_dur).gpsSeconds
    nsec = to_gps(gstop_or_dur).gpsNanoSeconds
    tend = str(sec) + '.' + str(nsec)

    if float(tend) < 1e6:
       gstop = gstart + float(tend)
    else:
       gstop = float(tend)

    # If the script is running on Virgo's server.
    if not remote:
        from virgotools import getChannel
        # Parse Virgo files
        with getChannel(source, channel, gstart, gstop, mask=mask, fill_value=fill_value) as data:
            data = TimeSeries(data.data, unit=data.unit, t0=gstart, dt=data.dt, channel=channel)
    else:
        # If not running the script on Virgo's server. Takes longer
        # Query is working, but crashes when computing q_transform.
        # Data might not be the same format as with PythonVirgoTools. 
        # Further checks required.

        if channel[:3] not in ['V1:', 'H1:', 'L1:']:
            print ('When accessing the data outside the virgo server, the channel must start with `V1:`, `H1:` or `L1:` ')
        data = TimeSeries.get(channel, gstart, gstop)
    return data

def Qtransform(data, qrange=qtransform.DEFAULT_QRANGE,
               frange=qtransform.DEFAULT_FRANGE, gps=None, search=0.5,
               tres="<default>", fres="<default>", logf=False,
               norm='median', mismatch=qtransform.DEFAULT_MISMATCH,
               outseg=None, whiten=True, fduration=2, highpass=None):
    """ Doctring from gwpy.timeseries.timeseries.q_transfrom 

        Scan a `TimeSeries` using the multi-Q transform and return an
        interpolated high-resolution spectrogram.

        By default, this method returns a high-resolution spectrogram in
        both time and frequency, which can result in a large memory
        footprint. If you know that you only need a subset of the output
        for, say, a figure, consider using ``outseg`` and the other
        keyword arguments to restrict the size of the returned data.

        Parameters
        ----------
        data : gwpy TimeSeries

        qrange : `tuple` of `float`, optional
            `(low, high)` range of Qs to scan
        frange : `tuple` of `float`, optional
            `(log, high)` range of frequencies to scan
        gps : `float`, optional
            central time of interest for determine loudest Q-plane
        search : `float`, optional
            window around `gps` in which to find peak energies, only
            used if `gps` is given
        tres : `float`, optional
            desired time resolution (seconds) of output `Spectrogram`,
            default is `abs(outseg) / 1000.`
        fres : `float`, `int`, `None`, optional
            desired frequency resolution (Hertz) of output `Spectrogram`,
            or, if ``logf=True``, the number of frequency samples;
            give `None` to skip this step and return the original resolution,
            default is 0.5 Hz or 500 frequency samples
        logf : `bool`, optional
            boolean switch to enable (`True`) or disable (`False`) use of
            log-sampled frequencies in the output `Spectrogram`,
            if `True` then `fres` is interpreted as a number of frequency
            samples, default: `False`
        norm : `bool`, `str`, optional
            whether to normalize the returned Q-transform output, or how,
            default: `True` (``'median'``), other options: `False`,
            ``'mean'``
        mismatch : `float`
            maximum allowed fractional mismatch between neighbouring tiles
        outseg : `~gwpy.segments.Segment`, optional
            GPS `[start, stop)` segment for output `Spectrogram`,
            default is the full duration of the input
        whiten : `bool`, `~gwpy.frequencyseries.FrequencySeries`, optional
            boolean switch to enable (`True`) or disable (`False`) data
            whitening, or an ASD `~gwpy.freqencyseries.FrequencySeries`
            with which to whiten the data
        fduration : `float`, optional
            duration (in seconds) of the time-domain FIR whitening filter,
            only used if `whiten` is not `False`, defaults to 2 seconds
        highpass : `float`, optional
            highpass corner frequency (in Hz) of the FIR whitening filter,
            used only if `whiten` is not `False`, default: `None`
        **asd_kw
            keyword arguments to pass to `TimeSeries.asd` to generate
            an ASD to use when whitening the data

        Returns
        -------
        out : `~gwpy.spectrogram.Spectrogram`
            output `Spectrogram` of normalised Q energy

        See also
        --------
        TimeSeries.asd
            for documentation on acceptable `**asd_kw`
        TimeSeries.whiten
            for documentation on how the whitening is done
        gwpy.signal.qtransform
            for code and documentation on how the Q-transform is implemented
    
    """
    q = data.q_transform(frange=frange, qrange=qrange, gps=gps,
                               search=search, tres=tres, fres=fres, logf=logf,
                               norm=norm, mismatch=mismatch, outseg=outseg,
                               whiten=whiten, fduration=fduration,
                               highpass=highpass)
    return q

def plot_spectro(Qtransform, gps_event, channel, add_chirp,
                 twindow=[0.5, 2, 10], m1=20, m2=10, s1z=0, s2z=0,
                 f_low=20, pn_2order=7, npoints=100,
                 approximant='SEOBNRv4', plot_ylim=[20,1400],
                 outDir='spectro.png',make_gif=False,outGif=None):
    """ Create spectrogramm plot"""

    # Get loudest Q-plane
    Q_loudest = Qtransform.q

    # get the time interval queried
    tmin = np.min(Qtransform.times)
    tmax = np.max(Qtransform.times)
    tmin = tmin.value
    tmax = tmax.value
    # Convert to gps times in seconds
    tevent_list = []
    for tevent in gps_event:
        sec = to_gps(tevent).gpsSeconds
        nsec = to_gps(tevent).gpsNanoSeconds
        t = str(sec) + '.' + str(nsec)
        tevent_list.append(float(t))
    # Compute the mean of the input gps_event. Used to center the spectrogram.
    tevent_mean = np.mean(tevent_list)

    # Compute the inspiral signal
    if add_chirp:
        track_t_list = []
        track_f_list = []
        for tevent in tevent_list:
            track_t, track_f = get_inspiral_tf(tevent, m1, m2, s1z, s2z,
                                               f_low, 
                                               #npoints=npoints,
                                               pn_2order=pn_2order,
                                               approximant=approximant)
            track_t_list.append(track_t)
            track_f_list.append(track_f)

    # Check whether the 'V1:' prefix is present in the input channel name
    if channel[:3] == 'V1:':
        channel_title = channel[3:]
    else:
        channel_title = channel

    if len(twindow) == 1:
        for wind in twindow:
            plot = plt.figure() 
            ax = plot.gca()
            if tevent_mean - wind < tmin:
                print ('The requested time window [%f, %f] starts before available data time interval [%f, %f]' % (tevent_mean-wind, tevent_mean+wind, tmin, tmax))
                print ('The x-axis of the spectrogram starts with first time available from the data.')
                xmin = tmin
            else: 
                xmin = tevent_mean - wind
            if tevent_mean + wind > tmax:
                print ('The requested time window [%f, %f] ends after available data time interval [%f, %f]' % (tevent_mean-wind, tevent_mean+wind, tmin, tmax))
                print ('The x-axis of the spectrogram ends with the latest time available from the data.')
                xmax = tmax
            else:
                xmax = tevent_mean + wind

            ax.imshow(Qtransform.crop(tevent_mean-wind,tevent_mean+wind))
            ax.set_yscale('log')
            ax.set_ylim(plot_ylim[0], plot_ylim[1])
            ax.set_ylabel('Frequency [Hz]')

            ax.set_xticks(np.linspace(tevent_mean - wind, tevent_mean + wind, 11))
            locs = ax.get_xticks()
            labels = [str(np.round(lab,1)) for lab in np.linspace(-wind,wind,11)]
            ax.set_xticklabels(labels)
            ax.set_xlim(xmin, xmax)
            ax.set_xlabel(r'Seconds from $t_0$')
            #ax.set_epoch(tevent_mean)
            ax.colorbar(clim=(0, 35), label='Normalised energy')

            if make_gif and add_chirp:

                import imageio
                import os

                #temporarily save plot without chirp
                plt.title('Omegascans of V1 (Q = %.1f): %s at %f' % (Q_loudest, channel_title, tevent_mean))
                #plt.tight_layout()
                plt.savefig("temp_save_no_chirp.png")

                #add chirp
                for tbank, fbank in zip(track_t_list, track_f_list):
                    ax.plot(tbank, fbank, lw=2, color='r')

                #temporarily save plot wit chirp
                #plt.tight_layout()
                plt.savefig("temp_save_with_chirp.png")
                
                images = []
                for filename in ["temp_save_no_chirp.png","temp_save_with_chirp.png"]:
                    images.append(imageio.imread(filename))
                imageio.mimsave(outGif, images,fps=0.5)

                #remove the temporare files
                os.remove("temp_save_no_chirp.png")
                os.remove("temp_save_with_chirp.png")


            if add_chirp:
                for tbank, fbank in zip(track_t_list, track_f_list):
                    ax.plot(tbank, fbank, lw=2, color='r')
            plt.title('Omegascans of V1 (Q = %.1f): %s at %f' % (Q_loudest, channel_title, tevent_mean))
            plt.tight_layout()
            plt.savefig(outDir)

    else:
        nb_window = len(twindow)
        fig, axes = plt.subplots(nrows=1, ncols=nb_window,
                                   figsize=(23, 5))

        for i, wind in enumerate(twindow):
            if tevent_mean - wind < tmin:
                print ('The requested time window [%f, %f] starts before available data time interval [%f, %f]' % (tevent_mean-wind, tevent_mean+wind, tmin, tmax))
                print ('The x-axis of the spectrogram starts with first time available from the data.')
                xmin = tmin
            else:
                xmin = tevent_mean - wind
            if tevent_mean + wind > tmax:
                print ('The requested time window [%f, %f] ends after available data time interval [%f, %f]' % (tevent_mean-wind, tevent_mean+wind, tmin, tmax))
                print ('The x-axis of the spectrogram ends with the latest time available from the data.')
                xmax = tmax
            else:
                xmax = tevent_mean + wind

            axes[i].imshow(Qtransform.crop(tevent_mean-wind,tevent_mean+wind))
            axes[i].set_yscale('log')
            axes[i].set_ylim(plot_ylim[0], plot_ylim[1])
            axes[i].set_ylabel('Frequency [Hz]')

            #axes[i].set_xscale('auto-gps')
            axes[i].set_xticks(np.linspace(tevent_mean - wind, tevent_mean + wind, 11))
            locs = axes[i].get_xticks()
            labels = [str(np.round(lab,1)) for lab in np.linspace(-wind,wind,11)]
            axes[i].set_xticklabels(labels)
            axes[i].set_xlim(xmin, xmax)
            axes[i].set_xlabel(r'Seconds from $t_0$')
            #axes[i].set_epoch(tevent_mean)
            axes[i].colorbar(clim=(0, 35), label='Normalised energy')


        if make_gif and add_chirp:

            import imageio
            import os

            #temporarily save plot without chirp
            plt.subplots_adjust(wspace=0.3)
            fig.suptitle('Omegascans of V1 (Q = %.1f): %s at %f' % (Q_loudest, channel_title, tevent_mean),
                         fontweight='bold')
            #plt.tight_layout()
            plt.savefig("temp_save_no_chirp.png")

        for i, wind in enumerate(twindow):
            if add_chirp:
                for tbank, fbank in zip(track_t_list, track_f_list):
                    axes[i].plot(tbank, fbank, lw=2, color='r')
            plt.subplots_adjust(wspace=0.3)

        if make_gif and add_chirp:
            #add chirp
            #for tbank, fbank in zip(track_t_list, track_f_list):
                #axes[i].plot(tbank, fbank, lw=2, color='r')

            #temporarily save plot wit chirp
            #plt.tight_layout()
            plt.savefig("temp_save_with_chirp.png")
            
            images = []
            for filename in ["temp_save_no_chirp.png","temp_save_with_chirp.png"]:
                images.append(imageio.imread(filename))
            imageio.mimsave(outGif, images,fps=0.5)

            #remove the temporare files
            os.remove("temp_save_no_chirp.png")
            os.remove("temp_save_with_chirp.png")

        fig.suptitle('Omegascans of V1 (Q = %.1f): %s at %f' % (Q_loudest, channel_title, tevent_mean),
                     fontweight='bold')
        #fig.tight_layout()
        plt.savefig(outDir)

    return True


if __name__ == "__main__":

   parser = argparse.ArgumentParser(
           description='Plotting spectrogram with best fitted template')

   parser.add_argument('--remote',
                       dest='remote',
                       action='store_true',
                       help='If the script is running outsie the Virgo server. By default, assuming it is ran on Virgo server.')

   parser.add_argument('--source',
                       dest='source',
                       type=str,
                       required=False,
                       default='raw',
                       help='Frame file, either a full path to a ffl or gwf file, or an abbreviation like `raw`, `trend`, which are looked up in the stancard location. If omitted, defaults to `raw`, but this default value is deprecated. Default: `raw`.')

   parser.add_argument('--channel',
                       dest='channel',
                       type=str,
                       required=False,
                       default='V1:Hrec_hoft_16384Hz',
                       help='Source datastream for these data. If missing, a prefix `V1:` is added. Default: `V1:Hrec_hoft_16384Hz`.')

   parser.add_argument('--tstart',
                       dest='tstart',
                       required=True,
                       help='GPS time start. Any input parsable by `~gwpy.time.to_gps` is fine.')

   parser.add_argument('--tend_or_dur',
                       dest='tend',
                       required=True,
                       help='GPS time end or duration in seconds if a float < 1e6  is provided. Any input parsable by `~gwpy.time.to_gps` is fine.')

   parser.add_argument('--mask',
                       dest='mask',
                       type=bool,
                       default=False,
                       help='If mask is False, missing samples will be replaced by fill_value. If it is True, the returned FrVect will have an attribute missing, which is a mask vector that is zero for missing samples. Default: `False`.') 

   parser.add_argument('--fill_value',
                       dest='fill_value',
                       default=np.nan,
                       help='Value that is used for missing samples if mask is False. Default: `np.nan`.')

   parser.add_argument('--gps_event',
                      dest='gps_event',
                      required=True,
                      nargs='+',
                      help='GPS time of the event. If multiple times are provided, the spectrogram is centered on the mean time, and the chirp is computed for each time.')

   parser.add_argument('--twindow',
                       dest='twindow',
                       nargs='+',
                       type=float,
                       default=[0.5, 2, 10],
                       help='The time window used for the spectrogram. The spectrogram time window is [`gps_event`-`twindow`,`gps_event`+`twindow`. If a list is provided, a spectrogram is created for each. Maximum number of windows: 3. Default: [0.5, 2, 10]')

   # Paramters defining the template signal to plot
   parser.add_argument('--no_chirp',
                       dest='no_chirp',
                       action='store_false',
                       help='If not adding the best template to the spectrogram.')

   parser.add_argument('--m1',
                       dest='m1',
                       required=True,
                       type=float,
                       help='Mass 1.')

   parser.add_argument('--m2',
                       dest='m2',
                       required=True,
                       type=float,
                       help='Mass 2.')

   parser.add_argument('--s1',
                       dest='s1',
                       required=True,
                       type=float,
                       help='Spin 1.')

   parser.add_argument('--s2',
                       dest='s2',
                       required=True,
                       type=float,
                       help='Spin 2.')

   parser.add_argument('--f_low',
                       dest='f_low',
                       required=False,
                       default=20,
                       type=float,
                       help='At which frequency the chirp is computed. gstlal starts at about 15, and pycbc at 18.5. Default: 20.')

   parser.add_argument('--pn_2order',
                       dest='pn_2order',
                       required=False,
                       default=7,
                       type=float,
                       help='pn 2order. Default: 7.')

   parser.add_argument('--npoints',
                       dest='npoints',
                       required=False,
                       default=100,
                       type=float,
                       help='Number of points to compute the time-frequency evolution of an inspiral signal. Default: 100.')

   parser.add_argument('--approximant',
                       dest='approximant',
                       required=False,
                       default='SEOBNRv4',
                       type=str,
                       choices=['TaylorF2', 'SPAtmplt', 'SEOBNRv2', 'SEOBNRv2_ROM_DoubleSpin', 'SEOBNRv2_ROM_DoubleSpin_HI', 'SEOBNRv4', 'SEOBNRv4_ROM'],
                       help='Approximant to be used. Note that the corresponding data file must be downloaded in the $LAL_DATA_PATH. Defautl: `SEOBNRv4`.')

   parser.add_argument('--plot_frange',
                       dest='plot_frange',
                       required=False,
                       type=float,
                       nargs=2,
                       default=[20,1400],
                       help='Spectrogram Y-axis limits, in Hz. Default: [20, 1400].')

   # Qtransform parameters. Migt no need all of them in the argparse.
   parser.add_argument('--qrange',
                       dest='qrange',
                       required=False,
                       type=float,
                       nargs=2,
                       default=[4,64],
                       help='range of Qs to scan. Default: [4, 64].')

   parser.add_argument('--frange',
                       dest='frange',
                       required=False,
                       type=float,
                       nargs=2,
                       help='range of frequencies to scan. Default: [0, inf].')

   parser.add_argument('--useGPS',
                       dest='useGPS',
                       required=False,
                       type=bool,
                       default=False,
                       help='Whether to use the central time, given by `gps_event`, of interest for determine loudest Q-plane. Default: False')

   parser.add_argument('--search',
                       dest='search',
                       required=False,
                       type=float,
                       default=0.5,
                       help='Window around `gps_event` in which to find peak energies, only used if `useGPS` is set to True')

   parser.add_argument('--tres',
                       dest='tres',
                       required=False,
                       type=float,
                       default=None,
                       help='Desired time resolution (seconds) of output `Spectrogram`. Default: abs(`outseg`) / 1000.')

   parser.add_argument('--fres',
                       dest='fres',
                       required=False,
                       type=float,
                       default=None,
                       help='Desired frequency resolution (Hertz) of output `Spectrogram`, or, if ``logf=True``, the number of frequency samples; give `None` to skip this step and return the original resolution. Default is 0.5 Hz or 500 frequency samples.')

   parser.add_argument('--logf',
                       dest='logf',
                       required=False,
                       type=bool,
                       default=False,
                       help='Boolean switch to enable (`True`) or disable (`False`) use of log-sampled frequencies in the output `Spectrogram`, if `True` then `fres` is interpreted as a number of frequency samples. Default: `False`')

   parser.add_argument('--norm',
                       dest='norm',
                       required=False,
                       default='median',
                       choices=['mean', 'median', False],
                       help='Whether to normalize the returned Q-transform output, and how. Default: `median`.')

   parser.add_argument('--mismatch',
                       dest='mismatch',
                       required=False,
                       type=float,
                       default=0.2,
                       help='Maximum allowed fractional mismatch between neighbouring tiles. Default: 0.2')

   parser.add_argument('--outseg',
                       dest='outseg',
                       required=False,
                       nargs=2,
                       default=None,
                       help='GPS `[start, stop)` segment for output `Spectrogram`, default is the full duration of the input.')

   parser.add_argument('--whiten',
                       dest='whiten',
                       required=False,
                       default=True,
                       help='Boolean switch to enable (`True`) or disable (`False`) data whitening, or an ASD `~gwpy.freqencyseries.FrequencySeries` with which to whiten the data. Default: True')
 
   parser.add_argument('--fduration',
                       dest='fduration',
                       required=False,
                       type=float,
                       default=2,
                       help='Duration (in seconds) of the time-domain FIR whitening filter, only used if `whiten` is not `False`. Default: 0.2')
 
   parser.add_argument('--highpass',
                       dest='highpass',
                       required=False,
                       type=float,
                       default=None,
                       help='Highpass corner frequency (in Hz) of the FIR whitening filter, used only if `whiten` is not `False`. Default: `None`')

   parser.add_argument('--outDir',
                       dest='outDir',
                       required=False,
                       type=str,
                       default='spectrogram.png',
                       help='Path to save images, including filename and extension. Default : current directory, `spectrogram.png`.')

   parser.add_argument('--verbose',
                       dest='verbose',
                       action='store_true',
                       help='Print all parameters values if used.')

   parser.add_argument('--make_gif',
                       dest='make_gif',
                       action='store_true',
                       help='If true save a .gif file at the same location than --outDir')

   parser.add_argument('--outGif',
                       dest='outGif',
                       type=str,
                       help='the output path of the .gif file')

   args = parser.parse_args()

   if args.verbose:
       print ('\nArguments values:')
       print ('-------------------')
       for key, value in vars(args).items():
           print ('%s : %s' % (key, value))
       print ('-------------------\n')

   if args.make_gif:
       if args.outGif is None:
           raise ValueError('You should specifie an output path using --outGif while requesting a .gif file')

   if args.frange:
       frange = args.frange
   else:
       frange = qtransform.DEFAULT_FRANGE

   if args.useGPS is True:
       # Compute the mean event time if a list was provided
       tevent_list = []
       for tevent in args.gps_event:
           sec = to_gps(tevent).gpsSeconds
           nsec = to_gps(tevent).gpsNanoSeconds
           t = str(sec) + '.' + str(nsec)
           tevent_list.append(float(t))
       # Compute the mean of the input gps_event. Used to center the spectrogram.
       tevent_mean = np.mean(tevent_list)
   else:
       tevent_mean = None

   # Transform the outseg into a gwpy Segments
   if args.outseg:
      outseg = Segment(float(args.outseg[0]), float(args.outseg[1])) 
   else:
      outseg = None

   #  Set fres and tres
   if args.fres is None:
       fres = "<default>"
   else:
       fres = args.fres
   if args.tres is None:
       tres = "<default>"
   else:
       tres = args.tres

   # Convert Virgo raw data in gwpy TimeSeries 
   try:
       data = read_virgo_timeseries(args.source, args.channel, args.tstart,
                                    args.tend, args.mask, args.fill_value,
                                    args.remote)
   except:
       print ('ERROR when querying the data.')
       tb = traceback.format_exc()
       print (tb)
       print ('Exit. No spectrogram named `%s` was produced. ' % args.outDir)
       data = None

   # Compute the multi-Qtransfrom
   if data:
       try: 
           Qtransform = Qtransform(data, qrange=args.qrange, frange=frange,
                                   gps=tevent_mean, search=args.search,
                                   tres=tres, fres=fres, logf=args.logf,
                                   norm=args.norm, mismatch=args.mismatch,
                                   outseg=outseg, whiten=args.whiten,
                                   fduration=args.fduration,
                                   highpass=args.highpass)
       except:
           print('ERROR when computing the Q-tranform.')
           tb = traceback.format_exc()
           print (tb)
           print ('Exit. No spectrogram named `%s` was produced. ' % args.outDir)
           Qtransform = None

   # Make spectrogram
   if Qtransform and data:
       try:
           plot_spectro(Qtransform, args.gps_event, args.channel,
                        args.no_chirp, twindow=args.twindow, m1=args.m1,
                        m2=args.m2, s1z=args.s1, s2z=args.s2,
                        f_low=args.f_low, pn_2order=args.pn_2order,
                        npoints=args.npoints, approximant=args.approximant,
                        plot_ylim=args.plot_frange, outDir=args.outDir, make_gif=args.make_gif,outGif=args.outGif)
           status = True
       except:
           print ('ERROR when generating the spectrogram plot.')
           tb = traceback.format_exc()
           print (tb)
           print ('Exit. No spectrogram named `%s` was produced. ' % args.outDir)
           status = False

   else: 
       status = False
