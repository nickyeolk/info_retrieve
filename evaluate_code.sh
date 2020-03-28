#!/bin/bash

app_name=$CI_COMMIT_REF_NAME

cat /proc/version

echo "Installing curl"
apt install -y curl

echo "Installing microsoft odbcsql related packages"
apt-get install -y gnupg2
apt-key adv --keyserver keyserver.ubuntu.com --recv-keys EB3E94ADBE1229CF
apt-get install -y apt-transport-https ca-certificates
curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add
curl https://packages.microsoft.com/config/ubuntu/16.04/prod.list > /etc/apt/sources.list.d/mssql-release.list

# echo "Updating libc6"
# apt-cache policy libc6
# apt-get install libc6

# echo "Installing build-essential"
# apt-get -y install --reinstall build-essential

echo "Updating apt-get"
apt-get update

conda update libgcc

echo "Installing msodbcsql17"
ACCEPT_EULA=Y apt-get -y install msodbcsql17

echo "Installing unixodbc"
apt-get -y install unixodbc unixodbc-dev

# echo "Installing sqlalchemy"
# apt-get install -y python-sqlalchemy

if [ -d "./tests" ]
then
    if [ ! -f "./requirements.txt" ]
    then
        echo "No requirements.txt file found"
        exit 1
    fi

    conda env update -n base --file "./environment.yml"
    # pip install --upgrade pip
    # pip install -r "./requirements.txt"
    # Manually put these back since conda env update removes them
    pip install pytest pylint radon

    pytest -x "./tests"

    if [ "$?" -gt "0" ]
    then
        echo "Test failed"
        exit 1
    fi

else
    echo "No tests"
fi
