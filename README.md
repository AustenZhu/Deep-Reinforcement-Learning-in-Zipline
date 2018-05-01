TODO: For some reason, when I wrote all this, I thought I would need to create a lot of different state space representations so I didn't use openai-gym, instead porting the spaces I needed over. A refactor using openai-gym should be done. 

To get set up: Create conda environment with python 3.5, and activate environment.
NOTE: You might need to install nbformat first - don't know why. 

```bash
conda env create -f freeze.yml -n <name_of_env>
source activate <name_of_env>
```
Once you're setup, you can run zip_learning.py or zip_learning_rrn.py to start training!
