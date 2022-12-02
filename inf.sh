while  read line
do 
    python inference.py $line

done < command_file.txt