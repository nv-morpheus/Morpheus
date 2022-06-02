#!/bin/bash

count=0

while "$@"; do
   (( count++ ));
   echo "Iteration ${count}"
done

echo "Ran ${count} times before failing"
