# MachineLearningWithPython
Starter files for understanding Machine Learning with Python



Retrieve an authentication token and authenticate your Docker client to your registry. Use the AWS CLI:

> aws ecr get-login-password --region ap-southeast-1 | docker login --username AWS --password-stdin 373035976448.dkr.ecr.ap-southeast-1.amazonaws.com

Note: If you receive an error using the AWS CLI, make sure that you have the latest version of the AWS CLI and Docker installed.


Build your Docker image using the following command. For information on building a Docker file from scratch see the instructions here . You can skip this step if your image is already built:

> docker build -t demo-predict-diabetes .


After the build completes, tag your image so you can push the image to this repository:

> docker tag demo-predict-diabetes:latest 373035976448.dkr.ecr.ap-southeast-1.amazonaws.com/demo-predict-diabetes:latest


Run the following command to push this image to your newly created AWS repository:

> docker push 373035976448.dkr.ecr.ap-southeast-1.amazonaws.com/demo-predict-diabetes:latest


Notes -> Replace account number with your AWS account.