old_hash="8344b292519b2a92a7c01773a7583dc7"
new_hash="d3222a0da433934a13f21480cffd9aff"

tm_dir="tm"

for file in $(ls $tm_dir | grep $old_hash); do
    new_file=$(echo $file | sed "s/$old_hash/$new_hash/")
    echo "cp $tm_dir/$file $tm_dir/$new_file"
    cp $tm_dir/$file $tm_dir/$new_file
done