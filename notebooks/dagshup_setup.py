import mlflow
import dagshub

mlflow.set_tracking_uri("https://dagshub.com/Aman-7745/mlops-project.mlflow")
dagshub.init(repo_owner='Aman-7745', repo_name='mlops-project', mlflow=True)



with mlflow.start_run():
  # Your dummy code here...
  mlflow.log_metric('accuracy', 42)
  mlflow.log_param('Param name', 'Value')
  