## Experiment tracking

ML-FLow

why its importent:
1. reproducibility
2. organization
3. optimization
    <br>1. 

Tracking experiment in spreadsheets?
why is not enought?
1. error prone
2. no standard format
3. visibility 

### MLFLOW:


"An open source platform for the machine learning lifecycle"

runs and artifacts:
1. backend store
    1. backend store persists MLflow entities (runs, parameters, metrics, tags, metadata etc..)
    2. mlflow entities can be inserted in mlrun.db
2. artifact store
    1. artifact store persists the files, models, images and in-memory objects.
    2. we could have artifacts store in remote http
    3. we can have artifacts store in local ./mlrun

1. tracking
2. models
3. model registry
4. projects

MLflow tracking modules allows you to organize your experiments into runs.
1. parameters -- path to training dataset, pre-processing, features kfold parameters etc..
2. metrics -- train, test, validation metrics.
3. metada -- algorithem, developer name for filtering.
4. artifacts -- visualization tracking, data etc.
5. models -- logging model

additional information:
1. source code
2. author, start and end date.




#### Model registry:
why needed:
1. what is the envinorment it should run? what code has changed? how do I load this model? Should I update hyperparameters in prod? etc..
2. how to role back to previous model
3. what data has changed ? preprocessing has changed?

mlflow tracking server --> mlflow model registry (staging, production and archive)
mlflow registry:
1. is a registry with taggig to different envinorments.
2. you need to make sure of size, time it took to train. etc..
3. it provides:
    1. Model lineage,
    2. model versioning
    3. stage transitions 
    4. annotations

