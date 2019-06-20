#!/usr/bin/bash

# initial setup
#sed -i 's/$/\,/' new_integrated_dataset_grouped.csv
#sed -i 's/\r//' new_integrated_dataset_grouped.csv
cp new_integrated_dataset.csv new_integrated_dataset_grouped.csv
rm organosilicons.csv
rm organometallics.csv
rm buffer.txt
rm finished.txt

# collect organosilicons into a file
touch organosilicons.csv
touch buffer.txt
while IFS="" read -r p || [ -n "$p" ]
do
  if [[ $p == *"Si"* ]]; then
    echo "$p" >> organosilicons.csv
    echo "$p,1" >> buffer.txt
  else
    echo "$p,0" >> buffer.txt
  fi
done < new_integrated_dataset.csv
sed -i 's/\r//' buffer.txt

# collect organometallics into a file
metallic_strs=(
  "bery"
  "Bery"
  "BERY"
  "Na" # but not "Nap"
  "Mg"
  "alu"
  "Alu"
  "ALU"
  "K+"
  "calc"
  "Calc"
  "CALC"
  "tit"
  "Tit"
  "TIT"
  "chromi"
  "Chromi"
  "CHROMI"
  "Mn"
  "Fe"
  "nick"
  "Nick"
  "NICK"
  "copp"
  "Copp"
  "COPP"
  "Zn"
  "gall"
  "Gall"
  "GALL"
  "Zr"
  "Nb"
  "Ag"
  "Cd"
  "Sn"
  "Hf"
  "tant"
  "Tant"
  "TANT"
  "Hg"
  "Tl"
  "Pb"
  "bor" # but not "born"
  "Bor"
  "BOR"
  "Si"
  "Ge"
  "ars"
  "Ars"
  "ARS"
  "Sb"
)
#echo ${metallic_strs[*]}
touch organometallics.csv
touch finished.txt
found=false
while IFS="" read -r p || [ -n "$p" ]
do
  found=false
  for i in "${metallic_strs[@]}"
  do
    if [[ $p == *"$i"* ]]; then
      if [[ $i == "Na" && $p == *"Nap"* ]]; then
        break
      fi
      if [[ ($i == "bor" || $i == "BOR" || $i == "Bor") && ($p == *"born"* || $p == *"BORN"* || $p == *"Born"*) ]]; then
        break
      fi
      echo "$p" >> organometallics.csv
      echo "$p,1" >> finished.txt
      found=true
      break
    fi
  done
  if [ "$found" = false ]; then
    echo "$p,0" >> finished.txt
  fi
#done < new_integrated_dataset.csv
done < buffer.txt
sed -i 's/\r//' finished.txt

# clean up
cp finished.txt new_integrated_dataset_grouped.csv
rm buffer.txt
rm finished.txt
