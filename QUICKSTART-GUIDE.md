# Quickstart Guide
This guide is meant for non-technical individuals who want to get hands on experience training and performing inference with ML models. This will not cover any engineering topics necessary to _construct_ this repo; rather it will focus on the lowest-friction way to _run_ this repo.

This guide makes the following assumptions:
1. You're a non-technical experimenter. 
2. You have a basic understanding of what it means to train and perform inference using a ML model.
3. You have a gmail and google drive account that is already setup. 
4. You'll use a Jupyter notebook in Google Colab (you don't need to know what those are yet).
5. [#4](https://github.com/mfranzon/yolo-training-template/issues/4) has been solved.

Follow the steps in the order listed

## Open Jupyter notebook in Colab
1. go to the notebook on github
2. change the URL
3. that should open the notebook in colab

## Colab Setup
1. GPU setup
2. Google Drive

## Run 1st training
1. Setup dataset and nc
2. Set epochs 
3. leave imgsz, batch, device, project, name all the same
4. runnit - see trained model in project dir

## OPTIONAL: Subsequent training (w/o starting over)
1. Update train_model() to point to trained model
2. setup dataset, nc, and epochs
3. runnit - updated model overwrites existing model 

## Run 1st inference
1. Setup input, and output_path
2. leave model_path and conf_thresh
3. runnit - see annotated image/video in output_path



