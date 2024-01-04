#!/bin/bash

data_path=${1:-"./data"}/t3p3
echo "Downloading and extracting to ${data_path}. You can pass another path as the first argument."
echo "data-path: ${data_path}"


# wget form gdrive as described in https://medium.com/@acpanjan/download-google-drive-files-using-wget-3c2c025a8b99
file_id=1yo9yWHCH3NACFbBa6I_qS1GMwDd-PCg5
mkdir -p ${data_path}
out_name=${data_path}/t3p3_data.tar.gz
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=${file_id}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${file_id}" -O ${out_name} && rm -rf /tmp/cookies.txt

# decompress
cd ${data_path}
tar -xzvf t3p3_data.tar.gz
