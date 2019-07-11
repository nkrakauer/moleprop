#!/usr/bin/bash

# initial setup
#sed -i 's/$/\,/' integrated_dataset_grouped.csv
#sed -i 's/\r//' integrated_dataset_grouped.csv
cp integrated_dataset.csv integrated_dataset_grouped.csv
rm organosilicons.csv
rm organometallics.csv
rm organotins.csv
rm acids.csv
rm buffer1.txt
rm buffer2.txt
rm buffer3.txt
rm finished.txt

# collect organosilicons into a file
touch organosilicons.csv
touch buffer1.txt
while IFS="" read -r p || [ -n "$p" ]
do
  if [[ $p == *"Si"* ]]; then
    echo "$p" >> organosilicons.csv
    echo "$p,1" >> buffer1.txt
  else
    echo "$p,0" >> buffer1.txt
  fi
done < integrated_dataset.csv
sed -i 's/\r//' buffer1.txt

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
touch buffer2.txt
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
      echo "$p,1" >> buffer2.txt
      found=true
      break
    fi
  done
  if [ "$found" = false ]; then
    echo "$p,0" >> buffer2.txt
  fi
#done < integrated_dataset.csv
done < buffer1.txt
sed -i 's/\r//' buffer2.txt

# collect organotins into a file
touch organotins.csv
touch buffer3.txt
while IFS="" read -r p || [ -n "$p" ]
do
  if [[ $p == *"Sn"* ]]; then
    echo "$p" >> organotins.csv
    echo "$p,1" >> buffer3.txt
  else
    echo "$p,0" >> buffer3.txt
  fi
done < buffer2.txt
sed -i 's/\r//' buffer3.txt

# collect acids into a file
touch acids.csv
touch finished.txt
while IFS="" read -r p || [ -n "$p" ]
do
  if [[ $p == *"acid"* ]]; then
    echo "$p" >> acids.csv
    echo "$p,1" >> finished.txt
  else
    echo "$p,0" >> finished.txt
  fi
done < buffer3.txt
sed -i 's/\r//' finished.txt

# clean up
cp finished.txt integrated_dataset_grouped.csv
rm buffer1.txt
rm buffer2.txt
rm buffer3.txt
rm finished.txt
