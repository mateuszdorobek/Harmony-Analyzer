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

Liczba pobranych utworów: 2137
Liczba utworów po augmentacji: 25644
Łączna liczba akordów po kolei 1243944
Średnia liczba akordów na utwór: 48.51
Liczba typów akordów: 62
Liczba dźwieków podstawowych akordu: 12
Liczba możliwych akordów (rozmiar słownika): 744


Analogies score - top 5 - 

Embedding Size Comparition - przy window_size=2:
MultihotEmbedding 0.002560819462227913 33
Word2Vec CBOW 0.28169014084507044 13
Word2Vec Skip-Gram 0.2752880921895006 14
FastText 0.31241997439180536 13

Embedding Window Comparition - dla najlepszych size z poprzedniego:
Word2Vec CBOW 0.28169014084507044 2
Word2Vec Skip-Gram 0.2791293213828425 2
FastText 0.30985915492957744 2


Najlepszy FastText 13, 2

root_change : 3.05%
chord_type_change : 16.8%
V-I_progression : 52.27%
II-V_progression : 100.0%
V/V-V_progression : 4.55%
less_common_progression : 2.33%