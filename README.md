# Eve Data Processing Pipeline

## Overview

The **Eve Data Processing Pipeline** is a high-performance, modular library designed to extract, deduplicate, clean, anonymize, and export large-scale Earth science and Earth observation datasets. It is part of the Earth Virtual Expert (EVE) initiative—an open-science program funded by the European Space Agency’s Φ-lab and developed by Pi School, in collaboration with Imperative Space and Mistral AI.

## Earth Virtual Expert (EVE)

**Earth Virtual Expert (EVE)** aims to advance the use of Large Language Models (LLMs) within the Earth Observation (EO) and Earth Science (ES) community.

- Website: https://eve.philab.esa.int/  
- Other repositories: https://github.com/eve-esa



## Features

1. Extraction Step - Extraction from different files formats like pdf, html and xml. You can pass in any heirarchy of folder structure.
2. Deduplication Step - Exact duplication and close duplication using lsh.
3. Cleaning Step - Handles all the irregularities and artifacts present in the documents. Performs latex equations and table correction using an LLM.
4. Pii Step - Aanonymizes the document by masking out the `names` and `emails` present in the documents.
4. Export Step - This steps saves all the processed files in the desired format.

## Getting started

1. Install the packages.

```
uv sync
```

2. Configure the `config.yaml` file. (Look at examples section on how to do this)

```
pipeline:
  batch_size: 10 # not applicable to dedup
  inputs:
    path: "input_dir"
  stages:
    - name: extraction
       config: { format: "xml"} # you can choose to specify the file format, if not the program auto-detects it
    - name: duplication
       config: { method: "lsh", shingle_size: 3, num_perm: 128, threshold: 0.8 }
    - name: pii
       config: { url: "http://127.0.0.1:8000" } # for the pii setup a presidio server like in server/pii_server.py
    - name: export # if you forget to add this step, its enabled on default and saves as md format
      config: { format: "md", destination: "output/files"}
```

3. Run the pipeline

```
eve run
```

4. For pdf based metadata extraction you need to setup [MonkeyOCR](https://github.com/Yuliang-Liu/MonkeyOCR/tree/main) in the `server` folder. 
   Run inference on the files and then setup metadata extraction config.yaml file.

## Examples

You can find examples of `config.py` files on how to use this pipeline effectively.

## Funding

This project is supported by the European Space Agency (ESA) through the Large Language Model for Earth Observation and Earth Science initiative, under the ESA Express Procurement Plus (EXPRO+) program. Contract: 4000144469/24/I-DT. Development is facilitated by ESA Φ-lab AI4EO as part of the Foresight Element within FutureEO Block 4.

## Citation 

If you use this project in academic or research settings, please cite:

## License

This project is released under the Apache 2.0 License - see the (LICENSE)[LICENSE] file for more details.

## Contributing

We welcome contributions!
Please open an issue or submit a pull request on GitHub to help improve the pipeline.