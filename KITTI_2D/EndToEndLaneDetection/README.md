# End to End Lane Detection

This repo is an unofficial implementation of the Paper 
![End-to-end-Lane-Detection](https://arxiv.org/pdf/1902.00293.pdf) by Wouter Van Gansbeke et al. 
For the official Implementation and analysis, please visit their [Github-Lane-Detection-Official] - Pytorch. 
I have taken a lot of inspiration from their repo as well as from 
[Lane-Detection-Unofficial-Implementation] - Tensorflow 1.12 Version 
to complete my repository with Tensorflow 2.3 version. 

# Dataset Description

Curated dataset was downloaded from the [Google Drive] of Wouter Van Gansbeke, which is a subset of 
[TuSimpe Lane Detection dataset] with pre-processing. Given below is a sample visualization of the dataset
from Curated Dataset using ![sample_viewer.py](sample_viewer.py)

![Sample 1](Figures/sample_visualization_1.png?raw=true)
![Sample 2](Figures/sample_visualization_2.png?raw=true)


[//]: #  (These are reference links used in the body of this note and get stripped out when the markdown processor 
does its job. There is no need to format nicely because it shouldn't be seen. 
Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

[End-to-end-Lane-Detection] : <https://arxiv.org/pdf/1902.00293.pdf>
[Github-Lane-Detection-Official] : <https://github.com/wvangansbeke/LaneDetection_End2End>
[Lane-Detection-Unofficial-Implementation]: <https://github.com/MaybeShewill-CV/lanenet-lane-detection>
[Google Drive]: <https://drive.google.com/drive/folders/1UECiIOGjIua9ORIDfcZft8XGTQ-iTzuD>
[TuSimpe Lane Detection dataset]: <https://github.com/TuSimple/tusimple-benchmark/issues/3>