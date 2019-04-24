# pyBK - Speaker diarization python system based on binary key speaker modelling

The system provided performs speaker diarization (speech segmentation and clustering in homogeneous speaker clusters) on a given list of audio files. It is based on the [binary key speaker modelling ](https://www.isca-speech.org/archive/archive_papers/interspeech_2010/i10_2118.pdf) technique. Thanks to the in-session training of a binary key background model (KBM), the system does not require any external training data, providing an easy to run and tune option for speaker diarization tasks. 

## Description

This implementation is based on that of [Delgado](https://ieeexplore.ieee.org/document/7268861), which is also available for [MATLAB](https://github.com/h-delgado/binary-key-diarizer). Besides the binary key related code, useful functions for a speaker diarization system pipeline are included. Extra details and functionalities were added, following our participation at [EURECOM](http://audio.eurecom.fr/) on the [Albayzin 2016 Speaker Diarization Evaluation](https://iberspeech2016.inesc-id.pt/index.php/albayzin-evaluation/) described [here](https://pdfs.semanticscholar.org/05eb/6b90ceac6ad5a6de3b54885c6b12e9c9c689.pdf), the [first DIHARD challenge](https://coml.lscp.ens.fr/dihard/2018/index.html), detailed in the [Interspeech 2018 paper](https://www.isca-speech.org/archive/Interspeech_2018/pdfs/2172.pdf), and the [IberSPEECH-RTVE Speaker Diarization Evaluation](http://iberspeech2018.talp.cat/index.php/speaker-diarization-challenge/), explained [here](https://www.isca-speech.org/archive/IberSPEECH_2018/pdfs/IberS18_AE-5_Patino.pdf).

## Installation
This code is written and tested in python 3.6 using conda. It relies on a few common packages to get things done:
* [numpy](https://www.numpy.org/)
* [scipy](https://www.scipy.org/)
* [scikit-learn](https://scikit-learn.org/)
* [librosa](https://librosa.github.io/) for audio processing and feature extraction
* [py-webrtvad](https://github.com/wiseman/py-webrtcvad) for voice activity detection

If you are using conda:
```bash
$ conda create -n pyBK python=3.6
$ source activate pyBK
$ conda install numpy
$ conda install -c conda-forge librosa
$ pip install webrtcvad
$ git clone https://github.com/josepatino/pyBK.git
```

## Example

Five files from the [SAIVT-BNEWS database](https://research.qut.edu.au/saivt/databases/saivt-bnews/) are included in order to test the system (all rights reserved to their respective owners).  These comprise audio files in wav format, speech activity detection (SAD) and unpartitioned evaluation map (UEM) files obtained from the references. For a quick run:
```bash
$ cd pyBK
$ python main.py
```

In the case of not finding UEM files, the complete audio content will be considered.
In the case of not finding VAD files, automatic VAD based in [py-webrtvad](https://github.com/wiseman/py-webrtcvad) will be applied. Automatic VAD may also be enforced in the config file. 

System configuration is provided in the form of an [INI configuration file](https://docs.python.org/3/library/configparser.html), and comments are provided in the example config.ini file. To use this system on your data create a config file of your own and run:
```bash
$ python main.py yourconfig.ini
```

Finally, a config file following our DIHARD submission is also included. Note that this configuration is meant to be used with [IIR-CQT Mel-frequency cepstral coefficients (ICMC)](https://ieeexplore.ieee.org/document/7846262) which can be replicated using MATLAB code available [here](http://audio.eurecom.fr/content/software). 


## Evaluation
The system will have generated a [RTTM](https://github.com/nryant/dscore#rttm) file which you can evaluate using the NIST md-eval script provided,
```bash
$ eval-tools/md-eval-v21.pl -c 0.25 -s out/[experiment_name].rttm -r eval-tools/reference.rttm
```
which should return a 5.32% diarization error rate (DER) using a standard 0.25s collar. By using the automatic VAD you should get a 10.04% DER. As per the DIHARD config file, when using ICMCs as features, this system returns a DER of 30.69% on the evaluation set, with a 0s collar.

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
