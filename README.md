# Eve Data Processing Pipeline
(Design Document - https://docs.google.com/document/d/13sbBslvo7HGYX7pooL8tkjwldvrkVOsFyIZRJVTWLZg/edit?usp=sharing)

## TO-DO

(check if there is something better alt than asyncio thread)
1. pipeline parallelism - files flow as and when completed instead of halting (check how this applies to the data duplication phase)
2. cleaning pipeline
4. pii extraction
5. i think metadata extraction also part of this pipeline?
6. implement edge cases where multiple data formats is passed to the duplication stage and other stages, currently it assumes a particular format.
