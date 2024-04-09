#!/bin/bash

# Copy all models in the tm/ folder that start with 3b0db6128fc831f0802554de993ecf83,
# and rename them to f41be5e3da5bbaca1c8b599dc99f7e63, keeping the different postfixes
for file in tm/d44c2f30ca5233129259bcbde5c37e2c*; do
    cp $file $(echo $file | sed 's/d44c2f30ca5233129259bcbde5c37e2c/71cfad1a8cb3d64b1256649b306d997d/')
done
