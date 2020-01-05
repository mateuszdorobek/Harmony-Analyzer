# Dane Treningowe:

```
songs_and_chords
	A Foggy Day,Gershwin George,Medium Swing,F,,0,0,,"(4, 4)","['F^7', 'Gbo7', 'G-7', 'C7', 'F^7', 'Dh7', 'G7', 'C7', 'F^7', 'C-7', ... ]
chords_string_rep_no_bass
	['C-7', 'F7', 'Bb^7', 'Db7', ... ]
chords_string_rep_no_bass_aug_12
	to samo ale zaugmentowane do 12 tonacji
components_no_bass
	[[3, 7, 10], [3, 7, 10], [4, 7, 10], ... ]
multihot_no_bass_12
	[[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0], [0, 0, ...
multihot_no_bass_full (33)
	[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0
```

# **Testowanie**:

Opis plików testowych

```
root_change test generation. 
	C7 D7 F7 G7
chord_type_change test generation.
	C-7 C^7 D-7 D^7
V-I_progression test generation.
	C7 F^7 D7 G^7
II-V_progression test generation.
	C-7 F7 D-7 G7
V/V-V_progression test generation.
	C7 F7 D7 G7
less_common_progression test generation.
	others with all keys randomly choosen 10%
```

```
Accuracy for Word2Vec:

root_change: 1.8% (1594/88572)
chord_type_change: 5.6% (27133/483120)
V-I_progression: 3.0% (4/132)
II-V_progression: 80.3% (106/132)
V/V-V_progression: 13.6% (18/132)
less_common_progression: 2.0% (9686/480288)

Total accuracy: 3.7% (38541/1052376)
```

```
Accuracy for FastText:

root_change: 3.0% (2700/88572)
chord_type_change: 21.8% (105189/483120)
V-I_progression: 23.5% (31/132)
II-V_progression: 79.5% (105/132)
V/V-V_progression: 0.0% (0/132)
less_common_progression: 3.3% (15820/480288)

Total accuracy: 11.8% (123845/1052376)
```