# **Testowanie**:

```
root_change test generation.
chord_type_change test generation.
V-I_progression test generation.
II-V_progression test generation.
V/V-V_progression test generation.
less_common_progression test generation.
```

```
Accuracy for Word2Vec:
Evaluating...
2020-01-04 21:33:22,919 : INFO : Evaluating word analogies for top 300000 words in the model on data/validation/test_chords.txt
2020-01-04 21:33:29,577 : INFO : root_change: 1.8% (1594/88572)
2020-01-04 21:34:06,986 : INFO : chord_type_change: 5.6% (27133/483120)
2020-01-04 21:34:07,002 : INFO : V-I_progression: 3.0% (4/132)
2020-01-04 21:34:07,014 : INFO : II-V_progression: 80.3% (106/132)
2020-01-04 21:34:07,025 : INFO : V/V-V_progression: 13.6% (18/132)
2020-01-04 21:34:43,909 : INFO : less_common_progression: 2.0% (9686/480288)
2020-01-04 21:34:43,958 : INFO : Quadruplets with out-of-vocabulary words: 3.2%
2020-01-04 21:34:43,959 : INFO : NB: analogies containing OOV words were skipped from evaluation! To change this behavior, use "dummy4unknown=True"
2020-01-04 21:34:43,960 : INFO : Total accuracy: 3.7% (38541/1052376)
```

```
Accuracy for FastText:
Evaluating...
2020-01-04 21:34:44,091 : INFO : Evaluating word analogies for top 300000 words in the model on data/validation/test_chords.txt
2020-01-04 21:34:51,077 : INFO : root_change: 3.0% (2700/88572)
2020-01-04 21:35:27,911 : INFO : chord_type_change: 21.8% (105189/483120)
2020-01-04 21:35:27,929 : INFO : V-I_progression: 23.5% (31/132)
2020-01-04 21:35:27,946 : INFO : II-V_progression: 79.5% (105/132)
2020-01-04 21:35:27,957 : INFO : V/V-V_progression: 0.0% (0/132)
2020-01-04 21:36:04,817 : INFO : less_common_progression: 3.3% (15820/480288)
2020-01-04 21:36:04,868 : INFO : Quadruplets with out-of-vocabulary words: 3.2%
2020-01-04 21:36:04,869 : INFO : NB: analogies containing OOV words were skipped from evaluation! To change this behavior, use "dummy4unknown=True"
2020-01-04 21:36:04,870 : INFO : Total accuracy: 11.8% (123845/1052376)
```