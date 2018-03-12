To setup db: 
Download fundamental data from simfin (fully detailed), then rename the csv to fundamentals.csv. Place csv in same directory as the folder of the repo. 

Get the bloomberg data from a terminal, and transform it by following the instructions in bloomberg_transform.py

Run create_all from the models.py file. 

Then run

```bash
python ingest_shares_outstanding.py
python ingest_fundamentals.py
```
