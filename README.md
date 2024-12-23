# Software-analytics-bug-triaging

# Download the models 
the models can be downloaded from the following link: https://1drv.ms/u/s!AjZv2fgaLt2FyJxIPbxRrrumUckCiQ?e=fJQMir
The folder should be unzipped and placed  in the `models` folder and the other in the `models_recent` folder respectively.
## Setup
It is to be noted that python3.11 or newer is necessary to run this tool.
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

Add project to the PythonPath. This is necessary to import the modules. Without command ran from the README will not work.
```shell
export PYTHONPATH="$PYTHONPATH:$PWD"
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

### Running the model on the server
A conda environment with the correct version of python is already present on the server.
It can be activated as follow:
```shell
eval "$(/home/SA24-G1/miniconda3/bin/conda shell.bash hook)"
```
It might be necessary to add the project to the PythonPath.

## Data 
The programs assume the presence of three folder to store the data and the models, respectively `data`, `models`, `models_recent` folders.

## Data scraping
### Issues scraping 

To scrape issues from github, run the following command. it will store the scraped data in `/data/raw_parsed_issues.json`. 

```shell
python3 ./src/scraper/issues_scraper.py --starting_page 0 --max_issue_number 220000
```
For convenience, given that the scraping takes a significant amount of time, the data is already stored 
in the repository in `/data/raw_parsed_issues_1.tar.gz` and `/data/raw_parsed_issues_2.tar.gz`.

### Issue compressing 
In order to store the raw scraped data on Github, we need both to split it and compress it.
```shell
python3 ./src/utils/data_splitter.py 
```
The data is already divided and compressed in the repository in `/data/raw_parsed_issues_1.tar.gz` and `/data/raw_parsed_issues_2.tar.gz`.

### Commits scraping
To start scrape commits from github, run the following command.
By default, it will store the scraped data in `/data/commits_per_user.csv`.

```shell
python3 ./src/scraper/commits_scraper.py --max_iterations_per_user 100
```
The data is already stored in the repository in `/data/commits_per_user.csv` for convenience.

## Data preprocessing
In order to train and use the model the raw issues data needs to be preprocessed. To do so, run the following command.
It will store the preprocessed data in `/data/cleaned_parsed.issues.tar.gz`.

```shell
python3 ./src/DataHandler.py
```

Also the pre-processed data is already stored in the repository in `/data/cleaned_parsed.issues.tar.gz`.

## Model
Every subsequent script takes the following arguments:
- `--only_recent_issues` : is a flag, that if present, will make the script use the recent issues data. default is `True`
- `--use_gpu` : If True, the model will use the GPU. default is `True`
- `--batch_size` : The batch size for the model, default is `16`
- `--epochs` : The number of epochs for the model, default is `5`
- `--lr` : The learning rate for the model, default is `2e-5`

### Model training 
To train the model, run the following command. It will store the trained model in the `/models` directory.
If the '--only_recent_issues' flag is passed the trained data will be saved in `/models_recnt`.
```shell
python3 ./src/model/model_train.py --only_recent_issues
```

The below command will train the model on the recent issues and all issues, and automatically test the accuracy of the models.
```shell
python3 ./src/model/Predictor.py 
```

> :warning: If the batch size is set too high the model will not fit in the GPU memory and the training will fail. Adjust the batch size accordingly.

### Model evaluation
To evaluate the model, run the following command. It will print the evaluation metrics.
```shell
python3 ./src/model/model_evaluation.py --only_recent_issues
```
### Model prediction
To predict the top 5 assignee for a given issue, run the following command. It will print the top 5 assignee.
To be noted that '--issue_id' is the id of the issue and not the issue number.
```shell
python3 ./src/model/model_predictor.py --issue_id 752417277 --top_n 5 --only_recent_issues
```

## Visualization
To visualize cluster of the data, run the following command.
```shell
python3 ./src/visualization/visualization.py
```
## UI 
To run the UI, run the following command on the console (do not run it in an IDE).
```shell
python3  ./src/tui/tui.py
```