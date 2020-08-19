#!/bin/bash

for i in `ls data/raw | grep csv | head -n1`; do
  head -n1 data/raw/$i;
done

for i in `ls data/raw | grep csv`; do
  tail -n+2 data/raw/$i;
done
