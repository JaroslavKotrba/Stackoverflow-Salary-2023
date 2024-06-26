# Stackoverflow-Salary-2023

To view the deployed app: [https://jaroslavkotrba.shinyapps.io/salary_2023](https://jaroslavkotrba.shinyapps.io/salary_2023)

## Installation

### Step 1: Create and activate Conda environment
Create a new Conda environment with Python 3.11:
```sh
conda deactivate
conda env create -f environment.yml
```

### Step 2: Get and process data
Download data from [Stack Overflow Survey](https://survey.stackoverflow.co/) and put into `data` folder as `survey_results_public_2023.csv` to create `survey_clean.csv` run:
```sh
python analysis.py
```

### Step 3: Google Sheets API

#### Service account

To set up a service account for Google Sheets integration, follow these steps:

1. Go to the [Google Cloud Console](https://console.cloud.google.com/apis/)

2. Enable the necessary APIs and services:
    - Click on "ENABLE APIS AND SERVICES"
    - Search for "Google Sheets API" and "Google Drive API"
    - Enable both APIs

3. Create credentials for your service account:
    - After enabling the APIs, click on "CREATE CREDENTIALS"
    - Select "Service account"
    - Provide a name (e.g., "Salary 2023 Shiny in Python")
    - Click "CREATE AND CONTINUE"
    - Select the "Editor" role
    - Click "CONTINUE"
    - Click "DONE"

#### Get keys

1. To get the keys for your service account:
    - In the Google Cloud Console, go to "APIs & services" and click on "Service accounts"
    - Click on the service account you created
    - Go to the "KEYS" tab
    - Click on "ADD KEY" and select "Create new key"
    - Choose "JSON" and click "CREATE"
    - Save the generated JSON key file in a folder named `credentials`

#### Add email to the folder

Add the service account email to the folder with restricted access:
- Example email: `salary-2023-shiny-in-python@shiny-salary-2023-in-python.iam.gserviceaccount.com`
