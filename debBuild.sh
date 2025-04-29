#!/bin/sh

TEMP_DIR=tmp

echo "Starting deb package build"

echo "Making temporary directory tree"
mkdir -p $TEMP_DIR
mkdir -p $TEMP_DIR/etc/systemd/system/ 
mkdir -p $TEMP_DIR/usr/local/bin/
mkdir -p $TEMP_DIR/DEBIAN

echo "Copy control file for DEBIAN/"
cp src/DEBIAN/control $TEMP_DIR/DEBIAN/

echo "Copy python into place"
cp src/trees.py $TEMP_DIR/usr/local/bin/

echo "Copy mysklearn into place"
cp src/mysklearn -r $TEMP_DIR/usr/local/bin/

echo "Copy input_data into place"
cp src/input_data -r $TEMP_DIR/usr/local/bin/

echo "Building deb file"
dpkg-deb --root-owner-group --build $TEMP_DIR
mv $TEMP_DIR.deb tree.deb


echo "Complete."
