all:
	python prepareData.py
	python train.py policyNet
	python train.py valueNet
