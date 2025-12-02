# Earth Observation Evaluation Tasks

This page provides a comprehensive overview of all available Earth Observation (EO) evaluation tasks in eve-evalkit. Each task is designed to assess different capabilities of language models in the Earth Observation domain.

## Quick Reference

| Task Name | Type | Dataset | Size | Primary Metrics |
|-----------|------|---------|------|-----------------|
| [MCQA Multiple Answer](#mcqa-multiple-answer) | Multiple Choice | [eve-esa/eve-is-mcqa](https://huggingface.co/datasets/eve-esa/eve-is-mcqa) | 432  | IoU, Accuracy |
| [MCQA Single Answer](#mcqa-single-answer) | Multiple Choice | [eve-esa/mcqa-single-answer](https://huggingface.co/datasets/eve-esa/mcqa-single-answer) | 1308 | Accuracy |
| [Open Ended](#open-ended) | Generation | [eve-esa/open-ended](https://huggingface.co/datasets/eve-esa/open-ended) | 1304 | LLM as Judge |
| [Open Ended with Context](#open-ended-with-context) | Generation | [eve-esa/open-ended-w-context](https://huggingface.co/datasets/eve-esa/open-ended-w-context) | 606  | LLM Judge |
| [Refusal](#refusal) | Generation | [eve-esa/refusal](https://huggingface.co/datasets/eve-esa/refusal) | 946  | LLM Judge |
| [Hallucination Detection](#hallucination-detection) | Classification | [eve-esa/hallucination-detection](https://huggingface.co/datasets/eve-esa/hallucination-detection) | 2996 | Accuracy, Precision, Recall, F1 |

---

## Detailed Task Descriptions


### MCQA Multiple Answer

**Task Name:** `is_mcqa` or `mcqa_multiple_answer`

**Description:**

EVE-mcqa-multiple-answers consists of multiple-choice questions from Imperative Space MOOC exams where questions may have one or more correct answers. Models must identify all correct options from an arbitrary number of choices, making this a challenging task that requires comprehensive understanding rather than simple fact recall.

**How to Call:**

```yaml
tasks:
  - name: is_mcqa
    num_fewshot: 2
    max_tokens: 10000
```

**Dataset:**

- **Source:** [eve-esa/eve-is-mcqa](https://huggingface.co/datasets/eve-esa/eve-is-mcqa)
- **Split:** train
- **Size:** 432 samples
- **Structure:** Each example contains a `Question`, `Answers` (list of correct labels), and `Choices` (list with labels and text)

**Evaluation Metrics:**

- **IoU (Intersection over Union)**: Measures partial correctness by calculating the overlap between predicted and correct answer sets. IoU = |Predicted ∩ Correct| / |Predicted ∪ Correct| (higher is better)
- **Accuracy (Exact Match)**: Binary score where 1.0 means the predicted answer set exactly matches the correct answer set (higher is better)

**Why It's Useful:**

This task tests a model's comprehensive understanding of EO concepts where multiple aspects or factors may be simultaneously correct. The IoU metric is particularly valuable as it rewards partially correct answers, providing a more nuanced evaluation than simple exact matching. This reflects real-world scenarios where partial knowledge is still valuable.

**Example:**

```python
{
  "Question": "Which bands of Sentinel-2 have 10m resolution?",
  "Answers": ["A", "B", "C"],
  "Choices": {
    "label": ["A", "B", "C", "D"],
    "text": ["B2 (Blue)", "B3 (Green)", "B4 (Red)", "B8 (NIR)"]
  }
}
```

---

### MCQA Single Answer

**Task Name:** `mcqa_single_answer`

**Description:**

EVE-mcqa-single-answer is a traditional multiple-choice dataset with exactly one correct answer per question. Models must identify the single best option from the provided choices, testing factual knowledge and reasoning abilities in the Earth Observation domain.

**How to Call:**

```yaml
tasks:
  - name: mcqa_single_answer
    num_fewshot: 2
    max_tokens: 10000
```

**Dataset:**

- **Source:** [eve-esa/mcqa-single-answer](https://huggingface.co/datasets/eve-esa/mcqa-single-answer)
- **Split:** train
- **Size:** ~1000 samples
- **Structure:** Each example contains a `question`, `choices` (list of answer texts), and `answer` (single letter indicating correct choice)

**Evaluation Metrics:**

- **Accuracy**: Percentage of questions answered correctly (higher is better)

**Why It's Useful:**

This task evaluates factual knowledge and reasoning abilities in scenarios where there is a single definitively correct answer. It's particularly useful for assessing fundamental EO concepts, terminology, and principles. The single-answer format reduces ambiguity and provides clear, interpretable results.

**Example:**

```python
{
  "question": "What is the spatial resolution of Sentinel-2's visible bands?",
  "choices": ["5 meters", "10 meters", "20 meters", "60 meters"],
  "answer": "B"  # "10 meters"
}
```

---

### Open Ended

**Task Name:** `open_ended`

**Description:**

EVE-open-ended is a collection of ~969 open-ended question-answer pairs focused on Earth Observation. The dataset covers a wide range of EO topics including satellite imagery analysis, remote sensing techniques, environmental monitoring, and LiDAR. Models must generate free-form responses demonstrating deep understanding without the constraints of multiple-choice formats.

**How to Call:**

```yaml
tasks:
  - name: open_ended
    num_fewshot: 5
    max_tokens: 40000
    judge_api_key: !ref judge_api_key
    judge_base_url: !ref judge_base_url
    judge_name: !ref judge_name
```

**Dataset:**

- **Source:** [eve-esa/open-ended](https://huggingface.co/datasets/eve-esa/open-ended)
- **Split:** train
- **Size:** ~969 samples
- **Structure:** Each example contains a `Question` and `Answer` (reference)

**Evaluation Metrics:**

- **LLM as Judge**: A judge model evaluates the quality, accuracy, and completeness of generated answers using strict fact-checking rules (0 = FAIL, 1 = PASS)
- Alternative metrics: BLEU, ROUGE, Cosine Similarity, BERTScore

**LLM Judge Evaluation Rules:**

1. **Contradiction Check**: Fails if the answer contains ANY fact contradicting the reference
2. **Relevance Check**: Fails if the answer omits ESSENTIAL technical facts from the reference
3. **Additive Information**: Additional correct information is acceptable if it doesn't contradict
4. **Focus on Substance**: Ignores style, length, and tolerates minor phrasing differences

**Why It's Useful:**

This task assesses a model's ability to explain concepts, provide detailed answers, and demonstrate deep understanding. It's essential for evaluating models intended for educational or explanatory applications in EO, where nuanced explanations and technical accuracy are paramount.

---

### Open Ended with Context

**Task Name:** `open_ended_w_context`

**Description:**

EVE-open-ended-w-context provides open-ended questions that must be answered using 1-3 accompanying context documents. This tests the model's ability to extract and synthesize information from reference materials, making it ideal for evaluating Retrieval-Augmented Generation (RAG) systems. Not all samples contain all three documents, requiring models to handle variable numbers of context documents gracefully.

**How to Call:**

```yaml
tasks:
  - name: open_ended_w_context
    num_fewshot: 5
    max_tokens: 40000
    judge_api_key: !ref judge_api_key
    judge_base_url: !ref judge_base_url
    judge_name: !ref judge_name
```

**Dataset:**

- **Source:** [eve-esa/open-ended-w-context](https://huggingface.co/datasets/eve-esa/open-ended-w-context)
- **Split:** train
- **Structure:** Each example contains a `Question`, `Answer`, and up to three context documents (`Doc 1`, `Doc 2`, `Doc 3`)

**Evaluation Metrics:**

- **LLM Judge**: Evaluates whether answers are grounded in the provided context and correctly answer the question (higher is better)
- Uses the same strict fact-checking evaluation rules as open-ended tasks

**Why It's Useful:**

This task evaluates retrieval-augmented generation (RAG) capabilities, testing whether models can accurately extract information from provided documents rather than relying solely on parametric knowledge. This is crucial for applications where answers must be grounded in specific documentation or data sources. It also tests the model's ability to distinguish between context-provided information and pre-trained knowledge.

**Example:**

```python
{
  "Question": "What is the spatial resolution of Sentinel-2's visible bands?",
  "Answer": "Sentinel-2's visible bands have a spatial resolution of 10 meters.",
  "Doc 1": "The Sentinel-2 mission comprises a constellation...",
  "Doc 2": "Sentinel-2 carries the Multi-Spectral Instrument (MSI)...",
  "Doc 3": ""  # May be empty
}
```

---

### Refusal

**Task Name:** `refusal`

**Description:**

EVE-Refusal tests whether language models can appropriately refuse to answer questions when the provided context does not contain sufficient information. The dataset presents questions alongside context documents that intentionally lack the necessary information to answer. A well-calibrated model should recognize this limitation and refuse to answer, rather than generating plausible but incorrect information.

**How to Call:**

```yaml
tasks:
  - name: refusal
    num_fewshot: 5
    max_tokens: 40000
    judge_api_key: !ref judge_api_key
    judge_base_url: !ref judge_base_url
    judge_name: !ref judge_name
```

**Dataset:**

- **Source:** [eve-esa/refusal](https://huggingface.co/datasets/eve-esa/refusal)
- **Split:** train
- **Structure:** Each example contains a `question` and `context` (insufficient for answering)
- **Expected Answer:** "I'm sorry, but the provided context does not contain enough information to answer that question."

**Evaluation Metrics:**

- **LLM Judge**: Evaluates whether the model appropriately refuses to answer or acknowledges insufficient information (higher is better)

**Expected Behavior:**

- Recognize when provided context lacks sufficient information
- Explicitly refuse to answer or state information is not available
- Avoid generating plausible-sounding but fabricated information
- Maintain accuracy and honesty over completeness

**Why It's Useful:**

This task tests a critical safety and reliability feature: the ability to recognize limitations and avoid generating potentially incorrect information when context is insufficient. This prevents hallucinations and ensures trustworthy behavior in production systems. It's particularly important for RAG systems and applications where factual accuracy is paramount.

---

### Hallucination Detection

**Task Name:** `hallucination_detection`

**Description:**

EVE-Hallucination is a specialized dataset for evaluating language models' tendency to hallucinate in the Earth Observation domain. Unlike typical QA datasets, this contains deliberately hallucinated answers with detailed annotations marking which portions of text are hallucinated. The task is to identify whether a given answer contains hallucinated (false or unsupported) information.

**How to Call:**

```yaml
tasks:
  - name: hallucination_detection
    num_fewshot: 0
    max_tokens: 100
    judge_api_key: !ref judge_api_key
    judge_base_url: !ref judge_base_url
    judge_name: !ref judge_name
```

**Dataset:**

- **Source:** [eve-esa/hallucination-detection](https://huggingface.co/datasets/eve-esa/hallucination-detection)
- **Split:** train
- **Structure:** Each example contains `Question`, `Answer` (with hallucinations), `Soft labels` (probabilistic spans), and `Hard labels` (definite spans)

**Evaluation Metrics:**

- **Accuracy**: Overall correctness of hallucination detection (higher is better)
- **Precision**: Ratio of correctly identified hallucinations to all predicted hallucinations (higher is better)
- **Recall**: Ratio of correctly identified hallucinations to all actual hallucinations (higher is better)
- **F1 Score**: Harmonic mean of precision and recall (higher is better)

**Task Levels:**

1. **Binary Detection**: Determine if answer contains any hallucinated information (yes/no)
2. **Hard Span Detection**: Identify exact character spans that are hallucinated
3. **Soft Span Detection**: Identify spans with confidence scores

**Why It's Useful:**

This task evaluates a model's ability to self-assess and identify unreliable or fabricated information in EO contexts. Models with strong hallucination detection capabilities are more trustworthy and can potentially be used to validate outputs from other systems. This is crucial for safety-critical applications like climate monitoring, disaster response, and environmental analysis.

**Example:**

```python
{
  "Question": "What is the spatial resolution of Sentinel-2's visible bands?",
  "Answer": "Sentinel-2's visible bands have a spatial resolution of 5 meters, making it the highest resolution freely available satellite.",
  "Hard labels": [[52, 60], [73, 127]]  # Character spans that are hallucinated
}
```

---


## Running Tasks

### Using Configuration File

Add tasks to your `evals.yaml`:

```yaml
constants:
  judge_api_key: your-judge-api-key
  judge_base_url: https://openrouter.ai/api/v1
  judge_name: mistralai/mistral-large-2411
  tasks:
    - name: eo_summarization
      num_fewshot: 0
      max_tokens: 20000
      judge_api_key: !ref judge_api_key
      judge_base_url: !ref judge_base_url
      judge_name: !ref judge_name
    - name: is_mcqa
      num_fewshot: 2
      max_tokens: 10000
    - name: hallucination_detection
      num_fewshot: 0
      max_tokens: 100

models:
  - name: your-model-name
    base_url: https://api.provider.com/v1/chat/completions
    api_key: your-api-key
    temperature: 0.1
    num_concurrent: 5
    tasks: !ref tasks

output_dir: evals_outputs
```

Then run:

```bash
python scripts/evaluate.py evals.yaml
```

### Direct Command Line

```bash
lm_eval --model openai-chat-completions \
        --model_args base_url=https://api.provider.com,model=model-name,num_concurrent=5 \
        --tasks {task_name} \
        --include tasks \
        --num_fewshot 0 \
        --output_path ./outputs \
        --log_samples \
        --apply_chat_template
```

For tasks using LLM-as-judge metrics, set environment variables:

```bash
export JUDGE_API_KEY=your-judge-api-key
export JUDGE_BASE_URL=https://api.provider.com/v1
export JUDGE_NAME=judge-model-name
```

---

## Task Selection Guide

Choose tasks based on your evaluation goals:

**Factual Knowledge:**
- `mcqa_single_answer` - Single correct answer questions
- `is_mcqa` - Multiple correct answers with partial credit
- `wiley_mcqa` - Standardized textbook questions

**Generation Quality:**
- `eo_summarization` - Abstractive summarization of technical content
- `open_ended` - Free-form explanatory answers

**Grounded Generation (RAG):**
- `open_ended_w_context` - Answer questions using provided documents
- `refusal` - Recognize when context is insufficient

**Reliability & Safety:**
- `hallucination_detection` - Identify fabricated information
- `refusal` - Avoid answering without sufficient information

**Comprehensive Evaluation:**
- Run all tasks for a complete assessment across different capabilities

---

## Evaluation Best Practices

1. **Use Few-Shot Examples**: Most tasks benefit from few-shot examples (typically 2-5) to demonstrate the expected format
2. **Set Appropriate Timeouts**: Some tasks require longer generation (e.g., summarization), so adjust timeouts accordingly
3. **Configure Judge Model**: For LLM-as-judge tasks, choose a capable judge model (e.g., GPT-4, Claude 3.5 Sonnet, Mistral Large)
4. **Log Samples**: Always use `--log_samples` to inspect individual predictions and understand model behavior
5. **Monitor Costs**: LLM-as-judge evaluation can be expensive; consider using smaller subsets for initial testing

---

## Additional Resources

- **Dataset Repository:** [https://huggingface.co/eve-esa](https://huggingface.co/eve-esa)
- **GitHub Repository:** [https://github.com/eve-esa/eve-evaluation](https://github.com/eve-esa/eve-evaluation)
- **LM Evaluation Harness:** [https://github.com/EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)

---

## Citation

If you use these tasks or datasets in your research, please cite:

```bibtex
@misc{eve2025,
  title={EVE: Earth Virtual Expert},
  author={ESA},
  year={2025},
  publisher={HuggingFace},
  url={https://huggingface.co/eve-esa/eve_v0.1}
}
```

For the underlying evaluation framework:

```bibtex
@software{eval-harness,
  author       = {Gao, Leo and others},
  title        = {A framework for few-shot language model evaluation},
  month        = sep,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {v0.0.1},
  doi          = {10.5281/zenodo.5371628},
  url          = {https://doi.org/10.5281/zenodo.5371628}
}
```