#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

data_path=${T3P3_PATH:-${SCRIPT_DIR}/../t3p3_data}
data_path=$(readlink -m ${data_path})

echo "Downloading and extracting to ${data_path}. You can pass a different download path as the first argument if you want."
echo "data-path: ${data_path}"

# wget from gdrive as described in https://medium.com/@acpanjan/download-google-drive-files-using-wget-3c2c025a8b99
# switch to gdown. Seems to be more reliable than wget
file_id=1dPysMHuSuKPVnWYtIVMqSVnpnoHbwBMH
mkdir -p ${data_path}
out_name=${data_path}/t3p3_data.tar.gz
gdown https://drive.google.com/uc?id=$file_id --output $out_name
# wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=${file_id}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${file_id}" -O ${out_name} && rm -rf /tmp/cookies.txt

# decompress
cd ${data_path}
tar -xzvf t3p3_data.tar.gz
