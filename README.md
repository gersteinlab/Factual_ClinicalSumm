# Aligning Factual Consistency for Clinical Studies Summarization through Reinforcement Learning


# Metrics

## Rouge, METEOR, BLEU
Github repository: [nlg-eval](https://github.com/Maluuba/nlg-eval)
Usage:

1. Set up as described in the repository readme.

2. This repository provides a unified way to compute a series of natural language generation evaluation metrics. Here is an example of how to use it:

```python
from nlgeval import compute_metrics
metrics_dict = compute_metrics(hypothesis='examples/hyp.txt',
                               references=['examples/ref1.txt', 'examples/ref2.txt'])
```
The `hypothesis` parameter is the file path to the generated text. If there are multiple references, they can be provided as a list. If there is only one reference, it can be directly provided as a file path.

The metrics will be printed together.

## QAFactEval
Github repository: [QAFactEval](https://github.com/salesforce/QAFactEval)
Usage: First modify the `kwargs` and metric-related parameters, then store the `input.txt` as a list `[input]`, and store the `output.txt` as a nested list `[[output]]` (I understand the input and output in the example as source document and summary, respectively). Here is an example code:

```python
from qafacteval import QAFactEval
kwargs = {"cuda_device": 0, "use_lerc_quip": True, \
        "verbose": True, "generation_batch_size": 32, \
        "answering_batch_size": 32, "lerc_batch_size": 8}

model_folder = "" # path to models downloaded with download_models.sh
metric = QAFactEval(
    lerc_quip_path=f"{model_folder}/quip-512-mocha",
    generation_model_path=f"{model_folder}/generation/model.tar.gz",
    answering_model_dir=f"{model_folder}/answering",
    lerc_model_path=f"{model_folder}/lerc/model.tar.gz",
    lerc_pretrained_model_path=f"{model_folder}/lerc/pretraining.tar.gz",
    **kwargs
)

results = metric.score_batch_qafacteval(["This is a source document"], [["This is a summary."]], return_qa_pairs=True)
score = results[0][0]['qa-eval']['lerc_quip']
```

## SUPERT
Github repository: [SUPERT](https://github.com/danieldeutsch/SUPERT)
Usage: Use `CorpusReader` to read the target folder `data/topic_1`, which contains three subfolders `input_docs`, `references`, and `summaries` for input, reference, and output, respectively. Here is an example code:

```python
from ref_free_metrics.supert import Supert
from utils.data_reader import CorpusReader

# read docs and summaries
reader = CorpusReader('data/topic_1')
source_docs = reader()
summaries = reader.readSummaries() 

# compute the Supert scores
supert = Supert(source_docs) 
scores = supert(summaries)
```

## BLANC
Github repository: [BLANC](https://github.com/PrimerAI/blanc)
Usage: When there are multiple files, store the files as a list in `documents` and `summaries`, and use `eval_pairs()` to run the evaluation. When there is only one file, store it as a string, and use `eval_once()` to run the evaluation. Here is an example code:

```python
from blanc import BlancHelp, BlancTune
blanc_help = BlancHelp()
blanc_tune = BlancTune(finetune_mask_evenly=False, show_progress_bar=False)
# Single document
document = "Jack drove his minivan to the bazaar to purchase milk and honey for his large family."
summary = "Jack bought milk and honey."
blanc_help.eval_once(document, summary)
blanc_tune.eval_once(document, summary)
# Multiple documents
documents = ["Jack drove his minivan to the bazaar to purchase milk and honey for his large family.", "As Jill started taking a walk in the park, she certainly noticed that the trees were extra green this year."]
summaries = ["Jack bought milk and honey.", "Jill saw green trees in the park."]
blanc_help.eval_pairs(documents, summaries)
```
The above code runs on CPU. To use GPU acceleration, use the following code:
```python
blanc_help = BlancHelp(device='cuda', inference_batch_size=128)
blanc_tune = BlancTune(device='cuda', inference_batch_size=24, finetune_mask_evenly=False, finetune_batch_size=24)
```

## QAEval
Github repository: [QAEval](https://github.com/danieldeutsch/sacrerouge/blob/master/doc/metrics/qaeval.md)
This one seems to require installing the library instead of cloning the repository, so I will include its readme here. Usage:

1. Install SacreROUGE and then run `pip install qaeval`.

2. Run `sacrerouge setup-metric qa-eval`.

3. Store the summaries as a list `[]`, and store the references as a nested list `[[]]`, such as `[summary1, summary2]`, `[[reference1], [reference2]]`. Finally, call the `qaeval.score_all()` function. Here is an example code:

```python
import json
from sacrerouge.metrics import QAEval

summary1 = 'Dan walked to the bakery this morning.'
reference1 = 'Dan went to buy scones earlier this morning.'

# This line will load the generation and answer models into memory, so it may take some time to complete.
qaeval = QAEval()

# To score an individual summary with a list of reference summaries. This example
# only uses 1 reference, so it is wrapped in a list.
scores = qaeval.score(summary1, [reference1])
print(scores)
{'qa-eval': {'em': 0.5, 'f1': 0.5}}

# To run batch scoring, use the score_all function and pass a list of summaries and
# a list of list of references. Again, each instance here only has 1 reference, so it is wrapped
# in a list
summary2 = 'Roger Federer beat Rafael Nadal yesterday.'
reference2 = 'Yesterday, Nadal lost to Federer'
# scores_list is a list of size 2. scores_list[0] is the scores for summary1, and scores_list[1] for summary2
scores_list = qaeval.score_all([summary1, summary2], [[reference1], [reference2]])

# If you want the QA pairs used to score the summaries returned, add the return_qa_pairs=True argument
# to any of the scoring methods. A tuple of size 2 will be returned. The first item is the scores
# like above. The second item are the QA pairs.
scores, qas = qaeval.score(summary2, [reference2], return_qa_pairs=True)

# qas[i][j] is the j-th QA pair for the i-th reference summary. The "probability" is the QA model's
# probability for the prediction. "null_probability" is its probability there is no answer.
print(json.dumps(qas[0][0], indent=2))
```

## QuestEval
Github repository: [QuestEval](https://github.com/danieldeutsch/sacrerouge/blob/master/doc/metrics/qaeval.md)
Usage: This one also requires installing the library. Please refer to the readme in the above link. The `hypothesis` should be a list of all the outputs, `sources` should be a list of all the inputs, and `list_references` should be a list of all the inputs. Finally, call `score = questeval.corpus_questeval(hypothesis, sources, list_references)`. Here is an example code:

```python
from questeval.questeval_metric import QuestEval
questeval = QuestEval(no_cuda=True)

source_1 = "Since 2000, the recipient of the Kate Greenaway medal has also been presented with the Colin Mears award to the value of 35000."
prediction_1 = "Since 2000, the winner of the Kate Greenaway medal has also been given to the Colin Mears award of the Kate Greenaway medal."
references_1 = [
    "Since 2000, the recipient of the Kate Greenaway Medal will also receive the Colin Mears Awad which worth 5000 pounds",
    "Since 2000, the recipient of the Kate Greenaway Medal has also been given the Colin Mears Award."
]

source_2 = "He is also a member of another Jungiery boyband 183 Club."
prediction_2 = "He also has another Jungiery Boyband 183 club."
references_2 = [
    "He's also a member of another Jungiery boyband, 183 Club.", 
    "He belonged to the Jungiery boyband 183 Club."
]

score = questeval.corpus_questeval(
    hypothesis=[prediction_1, prediction_2], 
    sources=[source_1, source_2],
    list_references=[references_1, references_2]
)

print(score)
```

## FactCC, DAE
see FactCC, Github repository: [FactCC](https://github.com/nargesam/factCC/tree/master)

## SummaC
Github repository: [SummaC](https://github.com/tingofurro/summac/)

Usage:
1. Install SummaC by running `pip install summac`.

2. Import the necessary modules and create instances of the SummaC models:

```python
from summac.model_summac import SummaCZS, SummaCConv

model_zs = SummaCZS(granularity="sentence", model_name="vitc", device="cpu") # If you have a GPU: switch to: device="cuda"
model_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device="cpu", start_file="default", agg="mean")
```

3. Prepare your document and summary as strings and pass them to the `score` method of the respective model:

```python
document = """Scientists are studying Mars to learn about the Red Planet and find landing sites for future missions.
One possible site, known as Arcadia Planitia, is covered in strange sinuous features.
The shapes could be signs that the area is actually made of glaciers, which are large masses of slow-moving ice.
Arcadia Planitia is in Mars' northern lowlands."""

summary1 = "There are strange shape patterns on Arcadia Planitia. The shapes could indicate the area might be made of glaciers. This makes Arcadia Planitia ideal for future missions."

score_zs1 = model_zs.score([document], [summary1])
score_conv1 = model_conv.score([document], [summary1])

print("[Summary 1] SummaCZS Score: %.3f; SummacConv score: %.3f" % (score_zs1["scores"][0], score_conv1["scores"][0]))
```

This will output the scores for the provided summary using the SummaCZS and SummaCConv models respectively.



I hope this helps! Let me know if you have any further questions.

# Citation

```
@inproceedings{tang-etal-2023-aligning,
    title = "Aligning Factual Consistency for Clinical Studies Summarization through Reinforcement Learning",
    author = "Tang, Xiangru  and
      Cohan, Arman  and
      Gerstein, Mark",
    booktitle = "Proceedings of the 5th Clinical Natural Language Processing Workshop",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.clinicalnlp-1.7",
    pages = "48--58",
    abstract = "In the rapidly evolving landscape of medical research, accurate and concise summarization of clinical studies is crucial to support evidence-based practice. This paper presents a novel approach to clinical studies summarization, leveraging reinforcement learning to enhance factual consistency and align with human annotator preferences. Our work focuses on two tasks: Conclusion Generation and Review Generation. We train a CONFIT summarization model that outperforms GPT-3 and previous state-of-the-art models on the same datasets and collects expert and crowd-worker annotations to evaluate the quality and factual consistency of the generated summaries. These annotations enable us to measure the correlation of various automatic metrics, including modern factual evaluation metrics like QAFactEval, with human-assessed factual consistency. By employing top-correlated metrics as objectives for a reinforcement learning model, we demonstrate improved factuality in generated summaries that are preferred by human annotators.",
}
Creative Commons License
```
