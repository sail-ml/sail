#!/bin/bash
set -e -u -x

function repair_wheel {
    wheel="$1"
    if ! auditwheel show "$wheel"; then
        echo "Skipping non-platform wheel $wheel"
    else
        auditwheel repair "$wheel" --plat "$PLAT" -w /io/wheelhouse/
    fi
}


# Install a system package required by our library
curl https://raw.githubusercontent.com/dvershinin/apt-get-centos/master/apt-get.sh -o /usr/local/bin/apt-get
chmod 0755 /usr/local/bin/apt-get

cd io

rm -rf build/*
rm -rf io/wheelhouse/*

./builder/install.sh
yum -y install blas-devel


cd ..
# Compile wheels
for PYBIN in /opt/python/*/bin; do
    if [[ "$PYBIN" == *"cp36"* || "$PYBIN" == *"cp37"* || "$PYBIN" == *"cp38"* ]]; then
    # if [[ "$PYBIN" != *"cp27"* ]]; then
        "${PYBIN}/pip" install -r /io/requirements.txt
        "${PYBIN}/pip" install -r /io/requirements-dev.txt
        # "${PYBIN}/python3" io/setup.py bdist_wheel sdist -d wheelhouse/
        "${PYBIN}/pip" wheel /io/ --no-deps -w wheelhouse/
    fi
done

ls wheelhouse/

#Bundle external shared libraries into the wheels
for whl in wheelhouse/*.whl; do
    repair_wheel "$whl"
done

# Install packages and test
# for PYBIN in /opt/python/*/bin/; do
#     "${PYBIN}/pip" install python-manylinux-demo --no-index -f /io/wheelhouse
#     # (cd "$HOME"; "${PYBIN}/nosetests" pymanylinuxdemo)
# done