#!/bin/bash
# Produces blog post markdown file from the given ipynb file

git_root_dir=$( git rev-parse --show-toplevel )
nb_dir="${git_root_dir}/_notebook"
input_filename="${1}"
input_basename="$( basename -s .ipynb "${input_filename}" )"
output_filename="$( date +%F )-${input_basename}.md"
input_filepath="${nb_dir}/${input_filename}"
output_filepath="${git_root_dir}/_posts/${output_filename}"

echo "Converting '${input_filepath}'."
jupyter nbconvert "${input_filepath}" --to markdown
mv "${nb_dir}/${input_basename}.md" "${output_filepath}.tmp"
grep -v '<IPython.core.display.Javascript object>' "${output_filepath}.tmp" > "${output_filepath}.noheader"
cat "${nb_dir}/${input_basename}.header" "${output_filepath}.noheader" > "${output_filepath}"
rm "${output_filepath}.tmp"
rm "${output_filepath}.noheader"
echo "Finished. Post produced at '${output_filepath}'."

