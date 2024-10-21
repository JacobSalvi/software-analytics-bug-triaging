# Software-analytics-bug-triaging

## Setup

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
## Data scraping 

### Issues scraping 
To scrape issues from github, run the following command. 
By default, it will store the scraped data in `/data/raw_parsed_issues.json`. 

```shell
python3 ./src/scraper/issues_scraper.py
```

#### Issue compressing 
In order to be store the raw scraped data on Github, we need both to split it amd compress it.
```shell
python3 ./src/utils/data_splitter.py
```

### Commits scraping
To start scrape commits from github, run the following command.
By default, it will store the scraped data in `/data/commits_per_user.csv`.

```shell
python3 ./src/scraper/commits_scraper.py
```

## Data preprocessing
in order to train and use the model the raw issues data needs to be preprocessed. To do so, run the following command.

```shell
python3 ./src/DataHandler.py
```