# Video QA Model

The repo provides a videoqa dataset and several baseline models. 

## Dataset

`./data/videos.json` contains the video information.  

```
    [{
        video_id: <video id>,
        video_title: <video title>,
        video_url: <video url>
        transcript: ["sent1", "sent2", ...],
        segments: [
            {
                title: "title of the segment"
                timestamps:
                    start: <start timestamp>,
                    end: <end timestamp>,
                sentence_indexes:
                    start: <start index>,
                    end_index: <end index>
            }
        ]
    }]
```


`./data/qa_dataset.json` contains the question answer pairs
```
data: [ 
    {
        video_id: <video id>,
        question: "question",
        answer_start: <start index>
        answer_end: <end index>
    }
]
```

## Models

#### Attentive LSTM 
The model assumes that the video transcript is pre-segmented. It uses Attentive LSTM model (which is proposed in ﻿Improved Representation Learning for Question Answer Matching, ACL 2016) to score the relationship between the query and each segment. 

To train, 

```
cd codes/attentivelstm
python train.py --datadir ../../data
```


#### BM25
This is BM25 baseline. It uses ElasticSearch's BM25 to score the relationship between the query and each segment.

To run the test code, 
```
cd codes/bm25
python main.py
```

#### IWAN
The model assumes that the video transcript is pre-segmented and that each segment has a title. The model is based on IWAN model (which is proposed in Inter-Weighted Alignment Network for Sentence Pair Modeling, EMNLP 2017). It computes the similarity between the query and each title. It also considers the transcript texts. 

To train, 

```
cd codes/iwanvar
python train.py --datadir ../../data
```


#### RASOR_SENT
The model makes no assumption on the video data. It receives the plain transcripts and automatically detects the segment boundaries given a query. Our model is inspired by the RASOR model proposed in Learning Recurrent Span Representations for Extractive Question Answering, ArXiv, 2016. Unlike the original model which defines the spans at the token-level, our model drives the span embeddings at the sentence-level. 

To train, 

```
cd codes/rasorsent
python train.py --datadir ../../data
```

We also provide a variant of RASOR_SENT which takes the segment points as an additional input. The additional input makes the problem easier by making the model consider only those segment points rather than all sentences. To train this version,

```
cd codes/rasorsent_hint
python train.py --datadir ../../data
``` 

