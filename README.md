# software-analytics-bug-triaging

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