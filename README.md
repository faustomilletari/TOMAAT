# TOMAAT [Server Side]
TOMAAT: Serve deep learning models over the cloud. *Client side* demo with slicer [older version] https://youtu.be/rWH5J3QdoNI 

## Getting started
* Execute `pip install tomaat`
* Refer to documentation and examples

## Disclaimer
This software is provided "as-it-is" without any guarantee of correct functionality or guarantee of quality.
No formal support for this software will be given to users. 
This repository and any other part of the TOMAAT project should not be used for medical purposes. 
In particular this software should not be used to make, support, gain evidence on and aid medical decisions, 
interventions or diagnoses. Any use of this software for finalities other than "having fun with it" is prohibited.
The privacy of the data is not guardanteed when TOMAAT is used and we should not be held responsible for 
data mis-use following transfer. Although we reccommend to users of TOMAAT to organize their services 
such that no data is permanently stored or held when users request predictions, we cannot guardantee that any data 
transferred to those remote services is not going to be misused, stored or manipulated. 
Use TOMAAT-slicer and TOMAAT responsibly.

## Architecture
In this repository you find TOMAAT. 
TOMAAT is written in python and contains three main parts. 
One is the core portion, where we provide the main functions that are used to by TOMAAT to wrap deep learning
models and make them available over the cloud, through HTTP protocol, using Klein. The second part is the code of the
model endpoint announcement service, which is provided such that users are able to run their own endpoint announcementservices
if they wish. The third part consists of framework specific implementation that serve both as an example of how to use TOMAAT
to instantiate prediction services and as a fully functional solutions capable of spinning up pre-saved models that are 
fairly standard.

A summary of the current architecture of TOMAAT is shown below:
![architecture](http://tomaat.cloud/images/architecture.png)
All communications between local and remote machines -- for service discovery and inference -- happen through HTTP protocol. 
Services are discovered by a GET request to the appropriate URL while images are segmented through a POST 
request containing JSON data. Each service has its own interface that is defined by the server administrator as a collection of standard elements. User can trasfer data to services in order to get predictions. 

The interfaces used for communication are specified in the following:

### Service discovery interface
After the GET request is made to the service discovery server url (for example http://tomaat.cloud:8001/discover) a JSON message is received. It contains a *list* of dictionaries containing the following fields:
* 'interface_url': the URL that the user can GET from to obtain the description of the server interface
* 'prediction_url': the URL that the user can POST to in order to obtain predictions 
* 'modality': the medical imaging modality
* 'anatomy': the anatomy
* 'task': the task
* 'name': the name of the prediction service
* 'description': a description of the prediction service
* 'SID': an unique identifier for a server which allows specific services to be included in workflows
* 'creation_time': the time of last service announcement to the announcement service

### Service interface
Analysis happens by making a POST request to the prediction server. The post request needs to contain the necessary fields. Each service requires different arguments and data to be supplied by the client. The type and field (in the POST request) of these arguments is specified in the interface description. The interface description can be obtained by a GET request to the 'interface_url' hosted by the prediction server. The response to the get request will be a list of dictionaries containing an arbitrary combination of the following elements:
* `{'type': 'volume', 'destination': field}`: instucts the client to build its interface such that the user can choose a volume, in MHA format, and place it in the field `field` of the POST request.
* `{'type': 'slider', 'destination': field, 'minimum': a, 'maximum': b}`: instucts the client to build its such that the user can choose a value from a fixed interval [a, b] which will be expected to be in the field `field` of the POST request.
* `{'type': 'checkbox', 'destination': field, 'text': UI_text }`: instructs the client to build an interface widget similar to a checkbox, to allow the user to pass a on/off type of variable which is expected to be in the field `field` of the POST request.
* `{'type': 'radiobutton', 'destination': field, 'text': UI_text , 'options': ['UI_option1', 'UI_option2']}`: instructs the client to spawn a UI element similar to a radio button which allows the user to choose among multiple options, which will be passed to the server in the POST field `field`.

### Prediction
The user can trigger prediction on his own data through a POST request containing the necessary expected fields to the remote server. The request should be made to the URL in the `prediction_url` field of the server description dictionary obtained from the announcement service. 
A service that expects the interface
```
[{'type': 'volume', 'destination': 'image'},
 {'type': 'radiobutton', 'destination': 'type', 'text': 'MRI sequence' , 'options': ['T1', 'T2']}]
```
will expect POST requests having fields `image` and `type`. The `image` field will need to be populated the content of a MHA file and the `type` field will need to contain either the string `T1` or the string `T2`.
POST request should be multipart. An example of client can be found at the URL https://github.com/faustomilletari/TOMAAT-Slicer

## Core

ToDo

## Endpoint announcement service

ToDo

## Framework specific examples/solutions
