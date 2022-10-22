#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge UiO test_copy script CSV data into one file.
"""
__author__  = "Mark Butterworth"
__version__ = "0.1.0 20221022"
__license__ = "MIT"

# Ver 0.1.0 20220927  Initial version

# MIT License

# Copyright (c) 2021 Mark Butterworth

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys
import argparse
import glob
import csv
import fileinput
import re
import base64

from pathlib import Path
from io import BytesIO


# from functools import reduce

import pandas as pd
import numpy as np

import matplotlib as mpl
mpl.use('Agg')   # Ensure a non-interactive backend is used.
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


DEBUG = 0
VERBOSE = 0
FORCE = False

DATEFORMAT = '%a %d-%b %H:%M'
WATERMARK = 'Produced by Hitachi Vantara'


###############################################################################

TAGPAT = re.compile('<.*?>') 

def savechart(df, columns, title, file, xtitle=None, ytitle=None, 
    kind=None, stacked=False, style=None, figsize=None, dpi=120, marker=None,
    scaledown=None, linestyle=None, alpha=None, ylim=None, yscale=None,
    secondcols=None, scaledown2=None, 
    linestyle2=None, marker2=None, alpha2=0.6, ylim2=None, yscale2=None):

    if isinstance(columns, set):
        columns = list(columns) # in case type is a set
    if isinstance(secondcols, set):
        secondcols = list(secondcols) # in case type is a set

    figsize = figsize if figsize else [15, 3]
    linestyle2 = linestyle2 if linestyle2 else (0, (1,.5))


    def scaleLongest(slist):  # Need to test/work on this to prove correct
        if len(slist) == 0: return 4
        longest = len(str(max(slist, key=lambda x: len(str(x)) )))
        if longest > 50:    return 1
        elif longest > 40:  return 2
        elif longest > 32:  return 3
        elif longest > 25:  return 4
        elif longest > 16:  return 5
        elif longest > 12:  return 6
        elif longest > 8:   return 6
        elif longest > 6:   return 8
        else:               return 10

    if df.empty:
        pathspec = df.attrs.get('pathspec', None)
        if not pathspec:
            pathspec = f'savechart title: {title}'
        raise SystemExit(f'ERROR: dataframe is empty: pathspec: {repr(pathspec)}')

    cols = columns if columns else df.columns
    kind = 'line' if kind == None else kind

    if scaledown:
        dfview = df[cols].divide(scaledown, fill_value=0)
    else:
        dfview = df[cols]

    if DEBUG: print('savechart dfview.head:', dfview.head())
    

    if not xtitle:
        starttime = dfview.index.min().strftime(DATEFORMAT)
        endtime = dfview.index.max().strftime(DATEFORMAT)
        xtitle = f'{starttime} to {endtime}'
    
    if DEBUG: print('Setting up plot and figure')
    plt.style.use('seaborn-v0_8-darkgrid')
    ncol = scaleLongest(cols)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Set the color map cycle:
    # cm1 = tuple( x + (1,) for x in plt.cm.tab10.colors ) # create a colormap with half the alpha
    # cm2 = tuple( x + (0.1,) for x in plt.cm.tab10.colors ) # create a colormap with half the alpha
    # cmapcycle = mpl.colors.ListedColormap(cm1 + cm2) # Append to the original cm
    # cmapcycle = mpl.colors.ListedColormap(plt.cm.tab10.colors + plt.cm.Pastel2.colors)
    cmapcycle = mpl.colors.ListedColormap(plt.cm.tab10.colors + (mpl.colors.to_rgb('lightsteelblue'),)) # 11x color map works well for top10 + others
    ax.set_prop_cycle(color = cmapcycle(np.linspace(0,1,len(cmapcycle.colors)))) # set the color cycler

    if DEBUG: print('Rendering plot')
    if kind in ('area'):
        plot = dfview.plot(figsize=figsize, ax=ax, title=title, kind=kind, stacked=stacked, alpha=alpha, style=style, x_compat=True)
    else:
        plot = dfview.plot(figsize=figsize, ax=ax, title=title, kind=kind, stacked=stacked, linestyle=linestyle, marker=marker, alpha=alpha, style=style, x_compat=True)
        lines = ax.get_lines()
        for i, line in enumerate(lines, -len(lines)):
            line.set_zorder(abs(i)) # changes plot order so that first items are on top


    # locator = mdates.AutoDateLocator(minticks=8, maxticks=12)
    # ax.xaxis.set_major_locator(locator)
    # print(mdates.AutoDateFormatter(ax.xaxis.get_majorticklocs()))

    # Sort out x-axis format:

    if DEBUG: print('Calculating x-axis limits')
    minutes = (dfview.index.max() - dfview.index.min()).total_seconds() / 60 
    majorloc = mdates.DayLocator()
    minorloc = mdates.AutoDateLocator(maxticks=24)
    majorfmt = mdates.DateFormatter('%H:%M')
    minorfmt = mdates.DateFormatter('%H:%M')

    if minutes > 1 and minutes <= 20:
        majorloc = mdates.HourLocator()
        minorloc = mdates.MinuteLocator()
    elif minutes <= 60:
        majorloc = mdates.HourLocator()
        minorloc = mdates.MinuteLocator(range(0, 60, 5))   # Do not use interval as it could fall outside of 
    elif minutes <= 2 * 60:
        majorloc = mdates.HourLocator()
        minorloc = mdates.MinuteLocator(range(0, 60, 10))
    elif minutes <= 4 * 60:
        majorloc = mdates.HourLocator()
        minorloc = mdates.MinuteLocator(range(0, 60, 15))
    elif minutes <= 8 * 60:
        majorloc = mdates.HourLocator()
        minorloc = mdates.MinuteLocator(range(0, 60, 30))
    elif minutes <= 24 * 60:
        majorloc = mdates.DayLocator()
        minorloc = mdates.HourLocator()
        majorfmt = mdates.DateFormatter('%a')
    elif minutes > 24 * 60 and minutes < 72 * 60:
        majorloc = mdates.DayLocator()
        minorloc = mdates.HourLocator(range(0, 24, 4))
        majorfmt = mdates.DateFormatter('%a')
    else:
        majorloc = mdates.AutoDateLocator(maxticks=6)
        minorloc = mdates.AutoDateLocator(maxticks=12)
        majorfmt = mdates.AutoDateFormatter(majorloc)
        minorfmt = mdates.AutoDateFormatter(minorloc)

    
    if DEBUG: print('Setting labels')
    if xtitle:
        ax.set_xlabel(xtitle, horizontalalignment='center')
    if ytitle:
        ax.set_ylabel(ytitle, verticalalignment='center')
    #plt.setp(ax.get_xticklabels(), rotation=0, horizontalalignment='center')

    if secondcols:
        if DEBUG: print('Setting second dfview2')
        if scaledown2:
            dfview2 = df[secondcols].divide(scaledown2, fill_value=0)
        else:
            dfview2 = df[secondcols]

        if DEBUG: print('savechart dfview2.head:', dfview2.head())
        
        if DEBUG: print('Setting second axis')
        ax2 = ax.twinx()
        ax2._get_lines.prop_cycler = ax._get_lines.prop_cycler #sets ax2 to use same ax color cycler
        if DEBUG: print('Rendering second plot')
        plot2 = dfview2.plot(ax=ax2, linestyle=linestyle2, marker=marker2, alpha=alpha2, x_compat=True)

        lines = ax.get_lines()
        for i, line in enumerate(lines, -len(lines)):
            line.set_zorder(abs(i)) # changes plot order so that first items are on top

        ax.get_legend().remove() # disable first legend
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels() # get the lines and labels for the second legend
        
        if DEBUG: print('Setting second legend & y-axis formats/locators')
        ax2.grid(False)
        lgd = ax2.legend(lines+lines2, labels+labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=ncol)
        if ylim2: ax2.set_ylim(ylim2)  # ylim must be passed as [min, max]
        if yscale: ax2.set_yscale(yscale)
        if yscale == 'linear' or yscale == None:
            ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(x, ',.1f').rstrip('0').rstrip('.'))) # 1 decimal place & strip trailing zeros

        if DEBUG: print(f'Setting second x-axis limits: {dfview2.index.min()=} {dfview2.index.max()=}')
        if dfview2.index.min() == dfview2.index.max():
            raise SystemExit(f'ERROR: xlimits are identical: {dfview2.index.min()} {dfview2.index.max()}')
        ax2.set_xlim(dfview2.index.min(), dfview2.index.max())

        # if DEBUG: print('Setting second x-axis format')
        # ax2.xaxis.set_major_locator(majorloc)
        # ax2.xaxis.set_minor_locator(minorloc)
        # ax2.fmt_xdata = majorfmt
        # ax2.xaxis.set_major_formatter(majorfmt)  # Must use x_compat=True above
    else:
        ax.tick_params(labeltop=False, labelright=True)
        if DEBUG: print(f'Setting Legend: {ncol=}')
        lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=ncol)

    if DEBUG: print(f'Setting x-axis limits: {dfview.index.min()} {dfview.index.max()}')
    if dfview.index.min() == dfview.index.max():
        raise SystemExit(f'ERROR: xlimits are identical: {dfview.index.min()=} {dfview.index.max()=}')
    ax.set_xlim(dfview.index.min(), dfview.index.max())
    if DEBUG: print('Setting x-axis formats')
    ax.xaxis.set_major_locator(majorloc)
    ax.xaxis.set_minor_locator(minorloc)
    # ax.fmt_xdata = majorfmt
    ax.xaxis.set_major_formatter(majorfmt)  # Must use x_compat=True above
    ax.xaxis.set_minor_formatter(minorfmt)  # Must use x_compat=True above

    if DEBUG: print('Setting autofmt_xdate')
    fig.autofmt_xdate(rotation=0, ha='center')  # Rotates and formats major date ticks
    
    if DEBUG: print('Setting y-axis ticker format')
    ax.grid(which='minor')
    if ylim: ax.set_ylim(ylim)  # ylim must be passed as [min, max]
    if yscale: ax.set_yscale(yscale)
    if yscale == 'linear' or yscale == None:
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.yaxis.set_minor_locator(ticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(x, ',.1f').rstrip('0').rstrip('.'))) # 1 decimal place & strip trailing zeros

    # if scaledown:
    #     if DEBUG: print('Setting y-axis ticker scaledown')
    #     ticks = ticker.FuncFormatter(lambda x, p: '{0:,.1f}'.format(x/scaledown, ',.1f').rstrip('0').rstrip('.')) # 1 decimal place & strip trailing zeros
    #     ax.yaxis.set_major_formatter(ticks)
    #     # loc = ticker.MultipleLocator(base=scaledown)
    #     # ax.yaxis.set_major_locator(loc)
    
    if DEBUG: print('Setting watermark')
    for pos, ha in ((0.01, 'left'), (0.5, 'center'), (0.99, 'right')):
        fig.text(pos, 0, WATERMARK,
                fontsize=7, color='grey', transform=ax.transAxes,
                ha=ha, va='bottom', alpha=0.4)
        fig.text(pos, 0.98, WATERMARK,
                fontsize=7, color='grey', transform=ax.transAxes,
                ha=ha, va='top', alpha=0.4)

    # Write to image file
    if type(file)==str and (file.lower().endswith('.png') or file.lower().endswith('.jpg')):
        if DEBUG: print(f'Saving figure to file: {file}')
        if VERBOSE: print(f'Saving {file}')
        #fig = plot.get_figure()
        fig.savefig(file, dpi=dpi, bbox_extra_artists=(lgd,), bbox_inches='tight')
    
    # Write to HTML
    if hasattr(file, 'write') or file.lower().endswith('.html'):
        img = BytesIO()
        if DEBUG: print('Saving png figure to buffer')
        fig.savefig(img, format='png', dpi=dpi, bbox_extra_artists=(lgd,), bbox_inches='tight')
        #fig.savefig(img, format='svg', dpi=120, bbox_extra_artists=(lgd,), bbox_inches='tight')
        img.seek(0)
        if DEBUG: print('Encoding image to base64')
        data_uri = base64.b64encode(img.read()).decode('utf-8')
        img_tag = f'<img src="data:image/png;base64,{data_uri}">'
        #img_tag = f'<img src="data:image/svg+xml;base64,{data_uri}">'
        if type(file)==str:
            if DEBUG: print(f'Appending figure to: {file}')
            with open(file,'a', encoding='utf-8') as fd:
                print(img_tag, file=fd)
        elif hasattr(file, 'write'):
            if DEBUG: print(f'Appending figure to: {file.name}')
            print(img_tag, file=file)
    if DEBUG: print('Closing figure')
    plt.close(fig)



def html2plain(rawhtml):
    """Remove HTML tags"""
    plaintext = re.sub(TAGPAT, '', rawhtml)
    return plaintext

def pre_process(iterable, fdout=None):
    """Pre_process Vdbench input lines"""
    for line in iterable:
        line = line.strip()   # Important as some files have trailing spaces on each line.
        # if line.startswith('<'):
        #     if fdout: fdout.write(html2plain(line))
        # else:
        yield(line)

def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.version = __version__
    parser.add_argument('-V', '--version', action='version')
    parser.add_argument('-D', '--debug', action='count',
        help='Increase debug level, e.g. -DDD = level 3.')
    parser.add_argument('-v', '--verbose', action='count',
        help='Increase verbose level, e.g. -vv = level 2.')
    parser.add_argument('-f', '--force', action='store_true',
         help='Overwrite *.vdb2.csv if it exists')
    parser.add_argument('filename', type=str,
        help='Output filename')
    parser.add_argument('csvfiles', type=str, nargs='+',
        help='Input filenames or "-" for stdin')

    args = parser.parse_args()
    global DEBUG, VERBOSE, FORCE
    if args.debug:
        DEBUG = args.debug
        print('Python version:', sys.version)
        print('DEBUG LEVEL:', args.debug)
        print('Arguments:', args)

    if sys.version_info < (3, 6):
        print('ERROR: Minimum required python version is 3.6')
        return 10

    VERBOSE = args.verbose
    FORCE = args.force

    outfile = Path(args.filename)
    outxls = outfile.with_suffix('.csv')
    outtot = outfile.with_suffix('.persec.csv')
    outplot = outfile.with_suffix('.mbps.html')

    if outxls.exists() and not FORCE:
        print(f'ERROR: output file already exists: {str(outxls)}')
        return 1
    if outtot.exists() and not FORCE:
        print(f'ERROR: output file already exists: {str(outtot)}')
        return 1
    if outplot.exists() and not FORCE:
        print(f'ERROR: output file already exists: {str(outplot)}')
        return 1


    rc = 0
    dflist = list()
    for spec in args.csvfiles:  # Needed in case wildcards are passed in windows
        files = ['-'] if spec == '-' else glob.glob(spec)   # checks for stdin "-"
        for fname in files:
            fname = Path(fname)
            with fileinput.input(fname) as fdin:
                
                # if str(fname) == '-':
                #     fdin = sys.stdin
                # else:
                #     fdin = open(fname, 'r')
                # csvin = csv.reader(pre_process(fdin), delimiter=' ', skipinitialspace=True)
                # for row in csvin:
                #     # do something with row
                #     pass
            
                print(f'Reading: {fname.name}')
                dflist.append(pd.read_csv(fname, header=0, names=['timestamp', 'mbcopied', 'filecount', 'mbps', 'mbnow', 'delcount']))
                # dflist[-1]['timestamp'] = pd.to_datetime(dflist[-1]['timestamp'])
                dflist[-1].set_index('timestamp', inplace=True)
                dflist[-1]['filename'] = fname.name
                # print(dflist[-1].info())
                
        if len(dflist) < 1:
            print(f'ERROR: no input csvfiles found.')
            return 2
        else:
            df = pd.concat(dflist)
            df.sort_index(inplace=True)
            # print(df)
            print(df.info())
            print(f'Saving {outxls.name}...')
            df.to_csv(outxls)   # Argh data too big for excel
            print(f'Saving {outtot.name}...')
            dftot = df.groupby(df.index)[['mbcopied', 'filecount', 'mbps', 'mbnow', 'delcount']].sum()
            dftot.to_csv(outtot)
            dftot.index = pd.to_datetime(dftot.index)
            print(f'Saving {outplot.name}...')
            fd = open(outplot, 'w')
            savechart(dftot, ['mbps'], 'Uio Test Script MiB/ Sec', file=fd)
            savechart(dftot, ['mbnow'], 'Uio Test Script MiB/ Sec', file=fd)


    return rc          


if __name__=='__main__':
    retcode = cli()
    exit(retcode)

