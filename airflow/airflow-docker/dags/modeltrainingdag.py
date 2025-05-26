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
    'ModelTraining',
    default_args=default_args,
    schedule='0 1 * * 7', 
    catchup=False,
    tags=['Train'],
) as dag:

    mobilenet = BashOperator(
        task_id='MobileNet',
        bash_command='python3 /opt/airflow/dags/trainMobile.py'
    )

    resnet = BashOperator(
        task_id='ResNet',
        bash_command='python3 /opt/airflow/dags/trainResNet.py'
    )

    CNN = BashOperator(
        task_id='CNN',
        bash_command='python3 /opt/airflow/dags/trainTradCNN.py'
    )

    end = EmptyOperator(task_id="all_done")

    mobilenet >> resnet >> CNN >> end
