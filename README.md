TODO: For some reason, when I wrote all this, I thought I would need to create a lot of different state space representations so I didn't use openai-gym, instead porting the spaces I needed over. A refactor using openai-gym should be done. 

To get set up: Create conda environment with python 3.5, and activate environment.
NOTE: You might need to install nbformat first - don't know why. 

```bash
conda env create -f freeze.yml -n <name_of_env>
source activate <name_of_env>
```

Afterwards, you'll need to get the database setup - the data is from simfin and bloomberg terminals. You can find my process for extracting and processing these data sources in zipdl/data/ingestion_stuff. If necessary, I can provide the completed database on request. 

Once you're setup, you can run zip_learning.py or zip_learning_rrn.py to start training!

Some quick results with a 2-factor model, sparse universe, with undertrained models, where all sortinos are calculated from a 2-week period:
![Train](https://raw.githubusercontent.com/AustenZhu/Deep-Reinforcement-Learning-in-Zipline/master/train_data.png)
![Train Sortino](https://raw.githubusercontent.com/AustenZhu/Deep-Reinforcement-Learning-in-Zipline/master/train_data_sortino.png)

Models vs Naive:

Final Returns Change
ddqn-0: 0.5332208041793293
ddqn-10: -0.04774836749946829
ddqn-20: 0.3311071280935572
ddqn-30: 0.0202970941744682
rrn-10: 0.059657398785356655
rrn-20: 0.0022288124954322397
rrn-30: -0.008694285065074837

Final Sortino change
ddqn-0: 0.685238838213208
ddqn-10: 0.062264615549842986
ddqn-20: 0.4883762817080995
ddqn-30: 0.04853887924525486
rrn-10: 0.16661150576597428
rrn-20: 0.022934633934009147
rrn-30: -0.085110228085691

Out of sample results:
![Test](https://raw.githubusercontent.com/AustenZhu/Deep-Reinforcement-Learning-in-Zipline/master/test_data.png)
![Test Sortino (not the best graph)](https://raw.githubusercontent.com/AustenZhu/Deep-Reinforcement-Learning-in-Zipline/master/test_data_sortino.png)

Final Returns Change
ddqn-0: 0.28547777910815264
ddqn-10: 0.188430768560971
ddqn-20: -0.09245480173715441
ddqn-30: 0.5129820513099714
rrn-10: 0.2947057350908498
rrn-20: -0.3586516169059371
rrn-30: -0.040198159647625485

Final Sortino Change
ddqn-0: 0.4190644828858481
ddqn-10: 0.2510847813219861
ddqn-20: -0.14976684370119905
ddqn-30: 0.6353655069088276
rrn-10: 0.4354968461587405
rrn-20: -0.3829120232139319
rrn-30: -0.007454717642829451

