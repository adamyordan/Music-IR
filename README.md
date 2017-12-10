# Music-IR
Information Retrieval system to search music through databases utilizing Information Retrieval techniques

# Dependencies
* Flask (web server)
* pandas (reading csv)
* tqdm (progress bar)

# Running
* download dataset from kaggle https://www.kaggle.com/gyani95/380000-lyrics-from-metrolyrics, create new "ir/data" directory, and put dataset in the directory
* create virtualenv for the project
```
virtualenv env		
```
* activate virtualenv
```
source env/bin/activate
```
* install depedencies
```
pip install -r requirements.txt
```

- Generate tf-idf by running preprocessing.py
```
python ir/preprocessing.py
```
- WIP