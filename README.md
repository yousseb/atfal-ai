# atfal-ai

Reproducing the output of https://github.com/BU-Spark/ml-atfal-mafkoda-missing-children pipeline with the following considerations in mind:

1. Use REST API in order to be able to offload work on multiple servers/worker nodes
2. Use CPU wherever possible to reduce cost

In order to do so, some networks will be replaced with lighter networks. OpenVino and onnx-runtime will be used wherever an implementation that provides similar quality is found.
