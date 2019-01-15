# pyBK - Speaker diarization python system based on binary key speaker modelling

## Description

The system provided performs speaker diarization (speech segmentation and clustering in homogeneous speaker clusters) on a given list of audio files. It is based on the [binary key speaker modelling ](https://www.isca-speech.org/archive/archive_papers/interspeech_2010/i10_2118.pdf) technique, and follows the implementation by [Delgado](https://ieeexplore.ieee.org/document/7268861), and available for [MATLAB](https://github.com/h-delgado/binary-key-diarizer). Extra details and functionalities are included following our participation at [EURECOM](http://audio.eurecom.fr/) on the [first DIHARD challenge](https://coml.lscp.ens.fr/dihard/2018/index.html), described in the [Interspeech 2018 paper](https://www.isca-speech.org/archive/Interspeech_2018/pdfs/2172.pdf).

## Installation
This code is written and tested in python 3.6 using conda. It relies on a few common packages to get things done:
* [numpy](https://www.numpy.org/)
* [scipy](https://www.scipy.org/)
* [scikit-learn](https://scikit-learn.org/)
* [librosa](https://librosa.github.io/) for audio processing and feature extraction

If you are using conda:
```bash
$ conda create -n pyBK python=3.6
$ source activate pyBK
$ conda install numpy
$ conda install librosa
$ git clone https://github.com/josepatino/pyBK.git
```

## Example

Five files from the [SAIVT-BNEWS database](https://research.qut.edu.au/saivt/databases/saivt-bnews/) are provided in order to test the system (all rights reserved to their respective owners).  These include audio files in wav format, speech activity detection (SAD) and unpartitioned evaluation map (UEM) files obtained from the references. For a quick run:
```bash
$ cd pyBK
$ python main.py
```
System configuration is provided in the form of an [INI configuration file](https://docs.python.org/3/library/configparser.html), and comments are provided in the example config.ini file. To use this system on your data create a config file of your own and run:
```bash
$ python main.py yourconfig.ini
```
## Evaluation
The system will have generated a [RTTM](https://github.com/nryant/dscore#rttm) file which you can evaluate using the NIST md-eval script provided,
```bash
$ eval-tools/md-eval-v21.pl -c 0.25 -s out/[experiment_name].rttm -r eval-tools/reference.rttm
``` 
which should return a 5.46% diarization error rate (DER).

## Contact

Please feel free to contact me for any questions related to this code:
- Jose Patino:       patino[at]eurecom[dot]fr

## Citation

If you use `pyBK` in your research, please use the following citation:

```bibtex
@inproceedings{patino2018,
  author = {Patino, Jose and Delgado, H{\'e}ctor and Evans, Nicholas},
  title = {{The EURECOM submission to the first DIHARD Challenge}},
  booktitle = {{Interspeech 2018, 19th Annual Conference of the International Speech Communication Association}},
  year = {2018},
  month = {September},
  address = {Hyderabad, India},
}
```
