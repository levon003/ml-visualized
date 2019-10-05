#!/bin/bash
# Produces blog post markdown file from the given ipynb file

git_root_dir=$( git rev-parse --show-toplevel )
nb_dir="${git_root_dir}/notebook"
input_filename="${1}"
output_filename="$( basename -s .ipynb "${input_filename}" ).md"
input_filepath="${nb_dir}/${input_filename}"
output_filepath="${git_root_dir}/posts/${output_filename}"

echo "Converting '${input_filepath}'."
jupyter nbconvert "${input_filepath}" --to markdown
mv "${nb_dir}/${output_filename}" "${output_filepath}.tmp"
grep -v '<IPython.core.display.Javascript object>' "${output_filepath}.tmp" > "${output_filepath}"
rm "${output_filepath}.tmp"
echo "Finished. Post produced at '${output_filepath}'."

