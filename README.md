# atfal-ai

Reproducing the output of https://github.com/BU-Spark/ml-atfal-mafkoda-missing-children pipeline with the following considerations in mind:

1. Use REST API in order to be able to offload work on multiple servers/worker nodes
2. Use CPU wherever possible to reduce cost

In order to do so, some networks will be replaced with lighter networks. OpenVino and onnx-runtime will be used wherever an implementation that provides similar quality is found.

License notes:
1. Wherever a third-party library is included due to the need to patch it for CPU, the code within the third-party library will follow the license model for that library.
2. Whevever a third-party code file is included, the code for that file will follow the original license model for that file. An example of this is the OpenVino demo files used in the pipeline for face detector and face enhancer.


   