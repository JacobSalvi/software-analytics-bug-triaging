import os

import requests
import json

username = 'norma1guy'
token = os.getenv("GITHUB_TOKEN")
header = {
    "Authorization" : f"token {token}",
    "Accept": "application/vnd.github.v3+json"
}

def get_closed_issues() :
    closed_issues = []
    page = 1
    per_page = 100

    url = f'https://api.github.com/repos/microsoft/vscode/issues?state=closed&page={page}&per_page={per_page}'
    response = requests.get(url,headers=header)
    while page <= 1000 :
        if response.status_code == 200 :
            issues = response.json()
            if issues :
                issues = [issue for issue in issues if 'pull_request' not in issue]
                closed_issues.extend(issues)
                page += 1
            else :
                break
        
    return closed_issues

def to_json(issues,filename = "closed_issues_1.json") :

    with open(filename,'w') as file :
        json.dump(issues,file,indent = 4)

    
    

        

closed = get_closed_issues()
to_json(closed)