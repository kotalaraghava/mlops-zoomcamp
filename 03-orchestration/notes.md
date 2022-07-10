
### Orchestration with prefect:
prefect: -- opensource and company

schedule and monitor work that you want to accomplish.

Negative engineering:

90% of engineering time spent.
1. retries when apis go down.
2. malformed data
3. notifications
4. observability into failure
5. 

#### Prefect:

If you mix and match perfect with native python then you should call .results to access the prefect code.
task:
    tasks is what prefect moniters
    it takes number of retries.
    function into task you get observability and logs are added for that.

flow:
    default is concurrent task runner.


parameters and type checking: you have automatic way of type checking so that no computation can be performed and stops execution instantly.

storage: you can store your flow so that while running it fecthces from it and run it
queue: you can create the work queue where we can pickup from storage and run the code.

next steps:
    we can integrate this with mlflow, where you can schedule the ml_training and check the how model is performing with current production model and if it's good then you can promote it to production.

    orchastration run this every week and then check for promotion using ml flow.



