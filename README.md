# Team 5: Inverse Cooking

This README file will guide you to install the necessary dependencies to run the software, and recreate the same results that we did. We will be using ASU's Agave Cluster to train this model.

## Installing Python 3.6 Dependencies

We are using Python 3.6. To install the correct dependencies, run the follow the command: 

`python3 -m pip install --user -r requirements.txt`



## Pulling the Inversecooking Repository from GitHub

***MAKE SURE TO DOWNLOAD EVERYTHING INTO SCRATCH SPACE***

Run this command to pull the Inversecooking repository from GitHub:

`git clone https://github.com/facebookresearch/inversecooking.git `

If you would like to test out their pretrained model, make sure to follow their directions, download the correct files (e.g `modelbest.ckpt`, `ingr_vocab.pkl`, `instr_vocab.pkl`), and place them in the correct directory. We will provide more directions on how to the pretrained model later in the README file.

## Downloading the Recipe1M Dataset

There are quite a few of `.tar` and `.json` files to download before getting started. The commands to download them will be listed below:

`wget http://wednesday.csail.mit.edu/temporal/release/recipe1M_layers.tar.gz`

`wget http://wednesday.csail.mit.edu/temporal/release/det_ingrs.json`

`wget http://wednesday.csail.mit.edu/temporal/release/recipe1M_images_train.tar`

`wget http://wednesday.csail.mit.edu/temporal/release/recipe1M_images_test.tar`

`wget http://wednesday.csail.mit.edu/temporal/release/recipe1M_images_val.tar`

These files should extracted using either command `tar xvf <file>.tar` or command `tar xvzf <file>.tar.gz` and the extracted files should be placed in the `inversecooking/data ` folder. The directory hierarchy should look like this:

```
\---data
	+---det_ingrs.json
		layers1.json
		layers2.json
		images
		+---train
			+---<training data>
		+---val
			+---<validation data>
		+---test
			+---<testing data>
```



## Running SBATCH Scripts 

To take advantage of Agave's resources, we need to be able to submit jobs through SLURM, a workload manager. We have three SBATCH scripts that needs to be run in this order: `build.sh`, `newdata.sh`, and `trainmodel.sh`. These scripts will be used to train the model with the **reduced** dataset and obtain some benchmarks for that model. Because our model has two parts, two separate trainings will occur and can be seen in the `trainmodel.sh` script. Before running these scripts, however, make sure to change the directories in the arguments to the directories in your directory. To run an SBATCH script, run the command `sbatch <file>.sh`. If there seems to be a problem in allocating a job to a node, you can tweak the parameters before the `python3` commands to find the best available node. To know which nodes are available, please use this link: https://rcstatus.asu.edu/agave/smallstatus.php. 



To run the pretrained model, do **not** use demo.ipynb as errors tend to pop up that way.  Instead, run the SBATCH script `pretrained.sh`. 



## How your output should look like

The output and error logs for these jobs will be located in `inversecooking/checkpoints/<model name>/logs`.  The training will output to files `train.log` and `train.err`. The benchmarking Python script will output to files `eval.log` and `eval.err`. You can compare your outputs to our files in the directory `logs` in our zip file. 

