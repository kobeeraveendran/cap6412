## Setup
First, make sure you have an up-to-date version of Anaconda on your system.

Start by using my provided `conda` environment by creating it from the file. Then, due to some mismatches between tensorflow versions I found in my case, install tensorflow-2.1.0 to be safe (otherwise, 1.15 may be fine).
```
conda env create -f environment.yaml
conda activate tf_keras

conda install ipython pip
pip install tensorflow
```

IMPORTANT: All datasets are expected to follow the following structure:
For Tiny Imagenet: keep `tiny-imagenet-200/` in the same directory as `utils.py` and `model_test.py`
For SVHN: create a directory called `svhn`, and move `train_32x32.mat` and `test_32x32.mat` into `svhn`.

At this point, attempt to run using the following command (stay in the root directory of the folder):

```
python model_test.py
```

I was not able to test this in a completely fresh environment, so in the event that dependency issues arise, install any other dependencies using the provided `requirements.txt` file, as seen below. I used CUDAv10.1.x with cuDNN 7.6.x, so I recommend also using that version if opting for this method.

(WARNING: this is from a `pip freeze` of my system's python packages. I have ensured my code works on my system (outside of a conda env), so this will likely also work for you, but there are many unnecessary packages as this was not run in a pipenv).

Note: I recommended doing this in a virtualenv to avoid interfering with any packages you may have on your system.
```
python3 -m pip install --user virtualenv

python3 -m venv env
source env/bin/activate

python3 -m pip install -r requirements.txt

python3 model_test.py

deactivate
```

## Running the code

Each part of the assignment can be tested by adjusting configurations set when running the code. Below are the config options you can set to fulfill the 3 parts of the assignment.

### Part 1
To select models to run (differing in the number of convolutional blocks they have), run the following command with a selected option (defaults to `conv3`).

```
python model_test.py --model conv3
```

Options for models are `conv3`, `conv6`, and `conv9`. These three models are all trained on Tiny Imagenet. Additionally, for part 3 of the assignment, you can run with `--model finetuned` to train the finetuned VGG16 on the SVHN dataset.

### Part 2
This part of the assignment tests performance of models on limited data. Each of the models in part 1 can be evaluated on varying amounts of training data using the following command:
```
python model_test.py --model <model> --samples_per_class 50
```
Where `samples_per_class` is a positive integer. The configurations are only applied to the Tiny Imagenet dataset, so this value defaults to 500, which is the maximum samples per class (the assignment tests a minimum of 50). NOTE: `samples_per_class` is ignored when running on the finetuned model, as it uses the full training data of SVHN.

### Part 3
As discussed earlier, you can finetune VGG16 pre-trained on Imagenet on the SVHN dataset by executing the following:
```
python model_test.py --model finetuned
```