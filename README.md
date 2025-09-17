# Eve Data Processing Pipeline

Eve data pipeline is a library to process, duplicate and clean data at a large scale. It has a set of steps that can extended further 


## Features

1. Extraction Step - It handles extraction from different files formats like pdf, html and xml. You can pass in nested folders and it handles multi-format folders.
2. Deduplication Step - It performs exact duplication and close duplication using lsh.
3. Cleaning Step - It performs a cleaning step to handle all the irregularities and artifacts present in the documents. Performs latex equations and table correction using an LLM.
4. Pii Step - It anonymizes the document and masks out the names and emails present in the documents.
4. Export Step - This steps saves all the processed files.

## Getting started

1. Install the packages.

```
pip install -e .
```

2. Configure the `config.yaml` file. (Look at examples section on how to do this)

```
pipeline:
  inputs:
    path: "input_dir"
  stages:
    - name: extraction
       config: { format: "xml"} # you can choose to specify the file format, if not the program auto-detects it
    - name: duplication
       config: { method: "lsh", shingle_size: 3, num_perm: 128, threshold: 0.8 }
    - name: pii
       config: { url: "http://127.0.0.1:8000" } # for the pii setup a presidio server like in server/pii_server.py
    - name: export
      config: { format: "md", destination: "output/files"}
```

3. Run the pipeline

```
eve run
```

## Examples

Example `config.py` files on how to use this pipeline.

## TO-DO


1. switch to uv
2. add an entrypoint to invoke the pipeline.
3. maybe nice to take in config file as an input
4. work on test cases.

