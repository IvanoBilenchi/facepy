#!/usr/bin/env bash
cd "$(dirname "${0}")"/..

source venv/bin/activate
python3 -m face_auth.main
deactivate
