#!/usr/bin/env bash
cd "$(dirname "${0}")"

# Safeguards
set -o pipefail
set -o errtrace
set -o errexit

echo "Cleaning up..."
rm -rf temp && mkdir temp

# Create virtualenv
if [ ! -d venv ]; then
	echo "Creating venv..."
	python3 -m venv venv
fi

echo "Installing Python package requirements..."
source venv/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt
deactivate

if [ ! -f res/dlib_face_recognition_resnet_model_v1.dat ]; then
	echo "Downloading models..."
	curl https://codeload.github.com/davisking/dlib-models/zip/master -o temp/models.zip

	echo "Extracting models..."
	unzip temp/models.zip -d temp
	bunzip2 temp/dlib-models-master/{dlib_face_recognition_resnet_model_v1.dat.bz2,mmod_human_face_detector.dat.bz2,shape_predictor_*_face_landmarks.dat.bz2}
	mv temp/dlib-models-master/*.dat res/
fi

if [ ! -d res/lfw ]; then
	echo "Downloading dataset..."
	curl http://vis-www.cs.umass.edu/lfw/lfw.tgz -o temp/lfw.tgz

	echo "Extracting dataset..."
	gunzip -c temp/lfw.tgz | tar xopf - -C temp
	mv temp/lfw res/
fi

rm -rf temp
echo "Done!"
