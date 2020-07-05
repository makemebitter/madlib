#!/bin/bash
set -e
root=${1:-"/local/madlib"}
hosts=${2:-"/local/gphost_list"}
# Make compile dir
cd "$root"
mkdir -p build
# Compile
cd build
cmake ..
make -j
# Copy to every host
readarray -t hostsarr < $hosts

for host in "${hostsarr[@]}"; do
    echo "Syncing $host ..."
    rsync -ar --update "$root/build" "gpadmin@$host:$root"
done

# Install
$root/build/src/bin/madpack -p greenplum -c gpadmin@master:5432/cerebro reinstall

