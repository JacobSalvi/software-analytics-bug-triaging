# Software-analytics-bug-triaging

## Setup
Run every script in the order given below.
### Virtual environment
Create virtual environment.
```shell
python3 -m venv .venv
```

Source virtual environment.
```shell
source .venv/bin/activate
```

Install requirements.
```shell
pip install -r requirements.txt
```

### Secrets
Create a .env file.
```shell
touch .env
```

Create a token on github and fill the .env as follows.
```shell
GITHUB_TOKEN=<TOKEN>
```

## Data 
The programs assume the presence of two folder to store the data and the model, respectively `data` and `models` folders.

## Data scraping 

### Issues scraping 
To scrape issues from github, run the following command. it will store the scraped data in `/data/raw_parsed_issues.json`. 

```shell
python3 ./src/scraper/issues_scraper.py
```
The data is already stored in the repository in `/data/raw_parsed_issues_1.tar.gz` and `/data/raw_parsed_issues_2.tar.gz`, so this step is not necessary.

### Issue compressing 
In order to be store the raw scraped data on Github, we need both to split it and compress it.
```shell
python3 ./src/utils/data_splitter.py
```
The data is already divided and compressed in the repository in `/data/raw_parsed_issues_1.tar.gz` and `/data/raw_parsed_issues_2.tar.gz`.

### Commits scraping
To start scrape commits from github, run the following command.
By default, it will store the scraped data in `/data/commits_per_user.csv`.

```shell
python3 ./src/scraper/commits_scraper.py
```
The data is already stored in the repository in `/data/commits_per_user.csv`.

## Data preprocessing
In order to train and use the model the raw issues data needs to be preprocessed. To do so, run the following command.
It will store the preprocessed data in `/data/cleaned_parsed.issues.tar.gz`.

```shell
python3 ./src/DataHandler.py
```

Also the pre-processed data is already stored in the repository in `/data/cleaned_parsed.issues.tar.gz`.

## Model
### Model training
To train the model, run the following command. It will store the trained model in `/models` folder.
```shell
python3 ./src/model/moderl_train.py --use_gpu True --batch_size 64
```
### Model evaluation
To evaluate the model, run the following command. It will print the evaluation metrics.
```shell
python3 ./src/model/model_evaluation.py --use_gpu True
```
### Model prediction
To predict the top 5 assignee for a given issue, run the following command. It will print the top 5 assignee.
```shell
python3 ./src/model/model_predictor.py --issue_id 752417277 --use_gpu True
```
