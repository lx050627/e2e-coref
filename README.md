# Character Identification on Dialogue with Neural Coreference Resolution

### Introduction
This repository is to accomplish a shared task in SemEval 2018 - Task 4: Character Identification on Multiparty Dialogues. Main references are listed in the following:

* [SemEval 2018 - Task 4: Character Identification on Multiparty Dialogues](https://competitions.codalab.org/competitions/17310)
* [End-to-end Neural Coreference Resolution](https://homes.cs.washington.edu/~kentonl/pub/lhlz-emnlp.2017.pdf)
  * A demo of the code can be found here: http://www.kentonl.com/e2e-coref.
* [Robust Coreference Resolution and Entity Linking on Dialogues: Character Identification on TV Show Transcripts](http://www.aclweb.org/anthology/K/K17/K17-1023.pdf)


### Requirements
* Python 2.7
  * TensorFlow 1.4.0
  * pyhocon (for parsing the configurations)
  * NLTK (for sentence splitting and tokenization in the demo)

### Setting Up

* Download pretrained word embeddings and build custom kernels by running `setup_all.sh`.
  * There are 3 platform-dependent ways to build custom TensorFlow kernels. Please comment/uncomment the appropriate lines in the script.
* Run one of the following:
  * To use the pretrained model only, run `setup_pretrained.sh`
  * To train your own models, run `setup_training.sh`
    * This assumes access to OntoNotes 5.0. Please edit the `ontonotes_path` variable.

## Training Instructions

#### Coreference Resolution
* Experiment configurations are found in `experiments.conf`
* Choose an experiment that you would like to run, e.g. `best`
* For a single-machine experiment, run the following two commands:
  * `python singleton.py <experiment>`
  * `python evaluator.py <experiment>`
* For a distributed multi-gpu experiment, edit the `cluster` property of the configuration and run the following commands:
  * `python parameter_server.py <experiment>`
  * `python worker.py <experiment>` (for every worker in your cluster)
  * `python evaluator.py <experiment>` (on the same machine as your first worker)
* Results are stored in the `logs` directory and can be viewed via TensorBoard.
* For final evaluation of the checkpoint with the maximum dev F1:
  * Run `python test_single.py <experiment>` for the single-model evaluation.
  * Run `python test_ensemble.py <experiment1> <experiment2> <experiment3>...` for the ensemble-model evaluation.

#### Entity Linking
* Prepare mention embedding data:
  * `python entity_linking_helper.py <experiment>`
* Train entity linking model:
  * `python entity_linking_train.py <experiment>`
* Test entity linkning model:
  * `python entity_linking_test.py <experiment>`

## Demo Instructions

* For the command-line demo with the pretrained model:
  * Run `python demo.py final`
* For the web demo with the pretrained model:
  * Run `python demo.py final 8080`
  * Edit the URL at the end of `docs/main.js` to point to the demo location, e.g. `localhost:8080`
  * Open `docs/index.html` in a web browser.
* To run the demo with other experiments, replace `final` with your configuration name.

## Contact
Aoxuan Li
Pu-Chin Chen
Xin Liu
Yutian Zhang
