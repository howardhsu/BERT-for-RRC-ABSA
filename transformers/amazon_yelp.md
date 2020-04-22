# BERT Post-Training for Cross-domain Review

This is a newly post-trained cross-domain review model.  You can download it [HERE](https://drive.google.com/file/d/1YbiI9W3acj4d9JbCbu_SmRjz_tNyShYV/view?usp=sharing).


## Environment
This model is trained under using the library of [hugging face transformers 2.4.1](https://github.com/huggingface/transformers) (it should work for 2.+).
This library has gone through SIGNIFICANT changes since our NAACL paper (which uses 0.4).
The code is not backward compatible with 0.4 but the weights are alignable.

## Preprocessing
The training corpus are preprocessed using `src/gen_pt_amazon_yelp.py`.  
You need to download [Amazon dataset's 5-core reviews with meta data](http://jmcauley.ucsd.edu/data/amazon/links.html) to `data/pt/kcore_5_review` and [Yelp dataset (2019 version)](https://www.yelp.com/dataset/challenge) to `data/pt/yelp`.

## Performance
We didn't fully evaluate the performance of this model yet.  
On `laptop`, the performance is closer to the domain-level `BERT-DK` in NAACL, but using much more corpus and much longer training time (that is why domain knowledge post-training is more feasible and efficient).

## Citation
If you find this work useful, please cite as following.
```
@inproceedings{xu_bert2019,
    title = "BERT Post-Training for Review Reading Comprehension and Aspect-based Sentiment Analysis",
    author = "Xu, Hu and Liu, Bing and Shu, Lei and Yu, Philip S.",
    booktitle = "Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics",
    month = "jun",
    year = "2019",
}
```
