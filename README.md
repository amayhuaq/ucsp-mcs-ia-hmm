# ucsp-mcs-ia-hmm
Hidden Markov Models

###Ejecución de HMM
```
python hmm_p1.py gene.train gene.counts gene.dev gene_dev.p1.out
python hmm_p2.py gene.train gene.counts gene.dev gene_dev.p2.out
python hmm_p3.py gene.train gene.counts gene.dev gene_dev.p3.out
```

###Evaluación del resultado
```
python eval_gene_tagger.py gene.key gene_dev.p1.out
python eval_gene_tagger.py gene.key gene_dev.p2.out
python eval_gene_tagger.py gene.key gene_dev.p3.out
```
