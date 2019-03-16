# Video QA Model


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