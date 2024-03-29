#!/bin/bash
set -x

apt-get update
apt-get -y install git rsync python3-pip python3-venv python3-sphinx python3-sphinx-rtd-theme python3-numpy

pwd ls -lah
export SOURCE_DATE_EPOCH=$(git log -1 --pretty=%ct)

###################
# INSTALL PACKAGE #
###################
export PIP_CACHE_DIR=~/.local/pip-cache
mkdir -p $PIP_CACHE_DIR
ls -l $PIP_CACHE_DI
python3 -m pip install wheel
python3 -m venv nmf
source nmf/bin/activate
python -m pip install numpy
python -m pip install cython
python -m pip install wheel
python -m pip install . --user --no-deps

##############
# BUILD DOCS #
##############

# Python Sphinx, configured with source/conf.py
# See https://www.sphinx-doc.org/
cd docs
make clean
make html

#######################
# Update GitHub Pages #
#######################
git config --global user.name "${GITHUB_ACTOR}"
git config --global user.email "${GITHUB_ACTOR}@users.noreply.github.com"

docroot=`mktemp -d`
rsync -av "build/html/" "${docroot}/"

pushd "${docroot}"

git init
git remote add deploy "https://token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"
git checkout -b gh-pages

# Adds .nojekyll file to the root to signal to GitHub that
# directories that start with an underscore (_) can remain
touch .nojekyll

# Add README
cat > README.md <<EOF
# README for the GitHub Pages Branch
This branch is simply a cache for the website served from https://nely-epfl.github.io/NeuroMechFly/,
and is  not intended to be viewed on github.com.
EOF

# Copy the resulting html pages built from Sphinx to the gh-pages branch
git add .

# Make a commit with changes and any new files
msg="Updating Docs for commit ${GITHUB_SHA} made on `date -d"@${SOURCE_DATE_EPOCH}" --iso-8601=seconds` from ${GITHUB_REF} by ${GITHUB_ACTOR}"
echo msg
git commit -am "${msg}"

# overwrite the contents of the gh-pages branch on our github.com repo
git push deploy gh-pages --force

popd # return to main repo sandbox root

# exit cleanly
exit 0
