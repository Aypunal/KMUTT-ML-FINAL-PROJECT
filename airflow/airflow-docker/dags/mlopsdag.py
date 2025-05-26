from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.bash import BashOperator
from datetime import datetime

# Define default arguments
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 3, 5),
    'retries': 0,
}

# Define DAG
with DAG(
    'poc_image_scrape',
    default_args=default_args,
    schedule='0 0 * * *', 
    catchup=False,
    tags=['poc_scrape_test'],
) as dag:

    # Task 1:
    real = BashOperator(
        task_id='real_scrape',
        bash_command='python3 /opt/airflow/dags/real_image.py'
    )

    # Task 2:
    ai = BashOperator(
        task_id='ai_scrape',
        bash_command='python3 /opt/airflow/dags/ai_image.py'
    )

    eda = BashOperator(
        task_id='eda',
        bash_command='python3 /opt/airflow/dags/eda.py'
    )
    # Define task dependencies
    real >> ai >> eda
