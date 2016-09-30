With the enormous amount of data surrounding us, it is important to be able to extract the most important information from it. In this notebook, we focus on one such information extraction algorithm from text. 

Broadly speaking, there are two different approaches for summarizing text: 1) extractive summarization, where the summary is lifted verbatim from the document itself and 2) abstractive summarization, where the summary is not a part of the document but is synthesized de novo by a learning model. 

Abstractive summarization is an extremely difficult problem and remains an area of active research. Some of the recent advancements employ recurrant neural networks and can be found in the works mentioned in [Quora](https://www.quora.com/Has-Deep-Learning-been-applied-to-automatic-text-summarization-successfully).

Here, I approach text summarization from an extractive viewpoint, tackling two specific questions: 
   * Given a document, which are the most important sentences?   
   * Given a document, what are the key words?  

To answer these questions, I implement TextRank - a graph-based algorithm that ranks text in a document based on the importance of the text. TextRank is analogous to Google's PageRank and was introduced by Mihalcea and Tarau in the paper [TextRank: Bringing Order into Texts](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf).

TextRank is an unsupervised learning algorithm and much simpler to implement as compared to abstractive summarization and yet yields good Recall-Oriented Understudy for Gisting Evaluation (ROUGE) scores.

<br></br>

<p align="center">
  <img src="https://github.com/spookyQubit/TextRank/blob/master/images/summary_1.png" width="495" height="350"/>
</p>


