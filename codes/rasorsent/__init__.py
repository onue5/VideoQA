"""
(Segmentation-based) Split the video into multiple segments and then score the similarity between the query and the
segment.

The model is based on RASOR model (Learning Recurrent Span Representations for Extractive Question Answering, ArXiv, 2016)
We extend the model to define the segments at the sentence-level rather than at the token level.
"""