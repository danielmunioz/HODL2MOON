# HODL2MOON🚀

**An AI-powered Sentiment indicator for "meme" and crypto communities**

Containing several models finetunned with data primarily extracted from reddit, aimed to better recognize common words used by modern "meme" traders and communities for improved predictions.

All the models contain a backbone based on the DistilBert Architecture, pre-trained using a masked language modeling objective on a collection of crypto-oriented data extracted from the aforementioned social network, then a multi layer perceptron (MLP) head is used to further finetune the models on popular sentiment analysis datasets (such as "financial phrasebank", "Stanford Sentiment Treebank", etc.), currently supporting __sentiment classification__ (meassuring by classes such as either negative or positive) and __sentiment regression__ (mesuring sentiment from 0: _negative_, all the way up to 1: _positive_).

<br/>

**E.g. Inference on Bitcoin data extracted from reddit communities through 2021**

<br/>

***For reference: 0=Negative, 1=Neutral, 2=Positive***

**Average sentiment from 2021/01/01 to 2021/05/01**

![inference_01_to_05](https://user-images.githubusercontent.com/47380745/160052570-564b75af-8b63-417f-8bf7-c2330d48c020.png)

**Average sentiment from 2021/05/02 to 2021/08/16**
![inference_05_to_08](https://user-images.githubusercontent.com/47380745/160056557-0ef780e9-26e7-4193-8dca-53146f752c5f.png)

