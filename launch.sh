#!

conda init

conda deactivate

conda remove --name goban-size-recognition-tool --all

NB_OF_OCCURRENCES_OF_ENVIRONMENT="$(conda env list | grep -c goban-size-recognition-tool)"
echo "${NB_OF_OCCURRENCES_OF_ENVIRONMENT}"

if NB_OF_OCCURRENCES_OF_ENVIRONMENT==0
then
    conda env create -f environment.yml 
fi

conda init

conda activate goban-size-recognition-tool

python my_script.py