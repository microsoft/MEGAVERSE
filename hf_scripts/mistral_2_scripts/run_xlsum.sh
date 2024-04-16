#!/bin/bash

arg=${1}

save_dir=${arg:-"results"}

echo $save_dir

for file in "hf_scripts/mistral_2_scripts/parameters.yaml" 
do 
    for lang in "english" "swahili" "tamil" "thai" "russian" "portuguese" "arabic" "hindi" "igbo" "indonesian" "urdu" "kyrgyz" "oromo" "amharic" "azerbaijani" "burmese" "chinese_simplified" "welsh" "kirundi" "hausa" "scottish_gaelic" "nepali" "pashto" "persian" "pidgin" "serbian_cyrillic" "serbian_latin" "sinhala" "somali" "tigrinya" "turkish" "ukrainian" "uzbek" "vietnamese" "yoruba"
    do
        python -m mega.eval_XLSUM ${lang} $file $save_dir
    done
done