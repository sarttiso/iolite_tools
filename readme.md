# Iolite Tools

This repository contains a package for facilitating processing of files exported from [Iolite 4](https://iolite.xyz/), including reduced data and log files. 

The [guide.ipynb](guide.ipynb) notebook provides a tutorial of how to use this package.

The code assumes some peculiarities based on how I organize my LASS experiments: experiments are tagged by the month when they occurred (in `yyyy-mm` format) and by their run number. If there are multiple experiments within a month, runs should consecutively increment throughout the month. Thus, individual zircons form aliquots in the database, and analyses then correspond to the instruments on which spots were run. Given that each zircon typically has a single spot, the analysis and aliquot names end up being very similar, although this need not always be the case.