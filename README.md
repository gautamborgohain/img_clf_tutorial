# Pytorch Image Classification on AWS Lambda using Zappa

App to classify images into cats and dogs -> *But deployed in Serverless*

## Project Structure

-  `app.py` : This is the main entrypoint of the app that specifies the routes to the endpoint

-  `zappa_settings.json` :  Zappa deployment config; [Zappa Docs](https://github.com/Miserlou/Zappa)

-  `dev_server.sh` :  A helper script to run a developement server on your machine using docker.

- `zappa.sh` : A helper script to call zappa commands in a docker image running zappa

-  `core` : dir containing the model and other modules

## Dev Server

Run `./dev_server.sh` on the project dir  
On your browser, go to `http://localhost:5000/apidocs`

## Deploy

- Set the `AWS_ACCESS_KEY_ID` & `AWS_SECRET_ACCESS_KEY` env vars to your creds
- Run `./zappa deploy dev` This will deploy the first time
- Run `./zappa update dev` Use Update command to update an already deployed App
- Run `./zappa status dev` To check the endpoint, error rate and other information of the app

*Note: The swagger api template that is available when deployed to Lambda does not correctly append  '/dev' to the hostname. So when you try to query from the doc you will get `{"message": "Forbidden"}`. You can test by correcting the `Curl` provided and just testing from terminal*  

## Requirements

Docker

All other dependencies are specified inside the docker images that are used.
The Dockerfile for those images can be modified and are here

## Why Docker for dev and for using Zappa?

So that the development can be consistent across different machines. I use a Mac as personal machine and at work I use an Ubuntu server. Using docker as a dev server allows me to switch
between the setups.

## Where is the `requirements.txt` file?

The standard requirements are set in the docker images, it is here. I set it at docker build step. To update the requirements, you will need to create new images, which you can be either adding to the conda env in the images or building the images with new requirements.txt

## Why Zappa?

Convenience

- Zappa bundles in a lot of features that are incredibly useful for cloud deployments, like the keep_warm schedule for larger packages, SQS, SNS triggers, async tasks.   
-  It has a shallow learning curve compared to some other serverless deployment utilities
- Flask app - Zappa allows development of an app for deployment anywhere not just lambda. This project, disregarding the `zappa_setting.json`, is just a flask app that can be deployed on K8S too.

## Why Lambda?

- Free if not used: The model is deployed, low use, low cost; no use, no cost
- Concurrency: Lambda scales up automatically. So you can have a highly available service with minimal engineering of clusters or managing batches for GPU inferencing etc.
