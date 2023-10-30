#!bin/bash


source $HOME/mambaforge/bin/activate
mamba activate mega_v3

sh mega/hf_models/scripts/run_indicxnli.sh
sh mega/hf_models/scripts/run_xnli_all.sh
sh mega/hf_models/scripts/run_xcopa_all.sh
sh mega/hf_models/scripts/run_qa_all.sh
sh mega/hf_models/scripts/run_pawsx_eval.sh
sh mega/hf_models/scripts/run_xstory.sh


