# TOMAAT [Server Side]
TOMAAT: Serve deep learning models over the cloud

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
![architecture](http://tomaat.cloud/images/architecture.jpg)
All communications between local and remote machines -- for service discovery and inference -- happen through HTTP protocol. 
Services are discovered by a GET request to the appropriate URL while images are segmented through a POST 
request containing JSON data. The interfaces used for communication are specified in the following:

### Service discovery interface
After the GET request is made to the service discovery server url (for example http://tomaat.cloud:8000/discovery) a JSON message is received. It contains:
* 'hosts': list of URL of inference services (endpoints)
* 'modalities': list of modalities the endpoints are capable of processing
* 'anatomies': list of anatomies the endpoints are capable of segmenting
* 'descriptions': list of endpoint descriptions (which method is used, which resolution, how fast it is etc...)

### Segmentation service interface
Segmentation happens by first dumping to disk the selected volume in a temporary folder in MHA format. At this point the volume will be re-read back into the TOMAAT-slicer extension and processed into a string through 'base64.encodestring'. At this point a JSON message will be created with the following fields:
* 'content_mha': string containing the data from the MHA file
* 'threshold': threshold to apply to the final segmentation result
* 'module_version': the version of the TOMAAT-slicer extension

## Core

ToDo

## Endpoint announcement service

ToDo

## Framework specific examples/solutions
