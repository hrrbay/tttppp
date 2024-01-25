#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "-------------------------------"
echo "* Starting training on \mu\ell-net"
echo "-------------------------------"
${SCRIPT_DIR}/train_muell.sh
echo "-------------------------------"
echo "* Starting training on R3D"
echo "-------------------------------"
${SCRIPT_DIR}/train_r3d.sh
echo "-------------------------------"
echo "* Starting training on Hiera"
echo "-------------------------------"
${SCRIPT_DIR}/train_hiera.sh
echo "-------------------------------"
echo "* Starting training on TTTransformer"
echo "-------------------------------"
${SCRIPT_DIR}/train_tttransformer.sh