# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 10:24:29 2024

@author: cedri
"""

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from Data_Engineering_Project_ETL import DataEngineeringProject

default_args = {
    'owner': 'your_name',
    'depends_on_past': False,
    'start_date': datetime(2024, 2, 5),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'your_dag_id',
    default_args=default_args,
    description='Your DAG description',
    schedule_interval=timedelta(days=1),  # Change this as needed
)

def fetch_data_from_api():
    project = DataEngineeringProject()
    data_api = project.fetch_data_from_api()
    if data_api:
        url_csv = data_api[2]["url"]
        data_csv = project.load_data_from_csv(url_csv)

        return data_csv
    else:
        return None

def process_data_function():
    project = DataEngineeringProject()
    data_csv = project.fetch_data_from_api()
    if data_csv:
        data_processed = project.process_data(data_csv)
        return data_processed
    else:
        return None

def save_cleaned_data_function():
    project = DataEngineeringProject()
    data_processed = project.process_data_function()
    if data_processed:
        project.save_cleaned_data(data_processed)
        print("Projet sauvegardé")

with dag:
    fetch_data_task = PythonOperator(
        task_id='fetch_data_from_api',
        python_callable=fetch_data_from_api,
    )

    process_data_task = PythonOperator(
        task_id='process_data',
        python_callable=process_data_function,
    )

    save_data_task = PythonOperator(
        task_id='save_cleaned_data',
        python_callable=save_cleaned_data_function,
    )

    # Define dependencies between tasks
    fetch_data_task >> process_data_task >> save_data_task

if __name__ == "__main__":
    dag.cli()
