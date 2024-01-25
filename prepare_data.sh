#!/bin/bash

# check if T3P3_PATH env variable is set write it to data_path otherwise use default

data_path=${1:-"./data"}/t3p3
if [ -z ${T3P3_PATH+x} ]; then
    echo "T3P3_PATH is unset. Using default path."
else
    echo "T3P3_PATH is set to '$T3P3_PATH'."
    data_path=$T3P3_PATH
fi

echo "Downloading and extracting to ${data_path}. You can pass a different download path as the first argument if you want."
echo "data-path: ${data_path}"


# wget form gdrive as described in https://medium.com/@acpanjan/download-google-drive-files-using-wget-3c2c025a8b99
# switch to gdown. Seems to be more reliable than wget
file_id=1dPysMHuSuKPVnWYtIVMqSVnpnoHbwBMH
mkdir -p ${data_path}
out_name=${data_path}/t3p3_data.tar.gz
gdown https://drive.google.com/uc?id=$file_id --output $out_name
# wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=${file_id}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${file_id}" -O ${out_name} && rm -rf /tmp/cookies.txt

# decompress
cd ${data_path}
tar -xzvf t3p3_data.tar.gz
