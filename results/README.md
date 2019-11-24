Źródła danych:

- [Suggestions-for-additions-or-changes-to-the-Main-Jazz-Playlist](https://www.irealb.com/forums/showthread.php?22620-Suggestions-for-additions-or-changes-to-the-Main-Jazz-Playlist)

- [Standards-Individual-Songs](https://www.irealb.com/forums/showthread.php?4522-Jazz-1350-Standards-Individual-Songs)

- [Dixieland-Trad-Playlists](https://www.irealb.com/forums/showthread.php?10591-Dixieland-Trad-Playlists)

- [Fusion-and-Smooth-Jazz](https://www.irealb.com/forums/showthread.php?210-Fusion-and-Smooth-Jazz)

- [Contemporary-Jazz](https://www.irealb.com/forums/showthread.php?204-Contemporary-Jazz)

- [Pat-Metheny-songs](https://www.irealb.com/forums/showthread.php?209-Pat-Metheny-songs)

- [Gypsy-Jazz](https://www.irealb.com/forums/showthread.php?215-Gypsy-Jazz)


Used Libraries:

- pyRealParser
- BeautifulSoup
- urllib
- pyChord
- music21

song url:

```
'irealb://%32%36%2D%32=%43%6F%6C%74%72%61%6E%65%20%4A%6F%68%6E==%4D%65%64%69%75%6D%20%55%70%20%53%77%69%6E%67=%46==%31%72%33%34%4C%62%4B%63%75%37%5A%4C%37%62%44%34%46%5E%37%20%5A%4C%37%46%20%37%2D%43%5A%4C%37%43%20%37%41%5E%5A%4C%37%45%20%37%5E%62%44%5A%4C%37%62%41%42%62%5E%37%20%34%54%5B%41%2A%20%37%5E%41%5A%41%37%4C%5A%44%5E%62%44%5A%4C%37%62%41%20%37%5E%46%5B%41%5D%2A%20%37%43%20%37%2D%47%5A%4C%37%47%20%37%2D%37%20%45%37%4C%20%37%5E%62%47%43%5B%42%2A%5D%2D%37%20%46%37%46%5A%4C%37%43%20%37%5E%41%5A%4C%37%45%20%5E%37%62%44%5A%4C%37%62%41%20%37%5E%62%42%5A%4C%5E%37%58%79%51%43%5A%4C%37%43%37%5E%62%44%7C%4C%5A%45%2D%37%41%7C%51%79%58%37%2D%62%45%7C%51%79%58%37%62%5E%42%5A%4C%37%46%20%37%5E%44%5A%4C%37%41%20%62%37%58%79%51%37%46%20%37%2D%42%5A%4C%37%46%2D%37%20%43%37%4C%37%43%20%37%5E%41%5A%4C%37%45%20%37%5E%44%62%5A%4C%37%62%41%20%37%5E%46%5B%41%2A%5D%20%5A%43%2D%37%20%47%7C%51%79%58%62%5E%37%20%41%62%37%4C%5A%44%62%5E%37%20%45%37%4C%5A%41%5E%37%20%43%37%4C%5A%46%5E%37%20%20%20%5A==%30=%30==='
```

parsed by [pyRealParser](https://pypi.org/project/pyRealParser/#description):

```
Parsed 26-2
<pyRealParser.pyRealParser.Tune object at 0x0000016D0181E4E0>
Title: 26-2
Composer: Coltrane John
Style: Medium Up Swing
Key: F
Transpose: None
Comp style: 0
BPM: 0
Repeats: None
Time signature: 4/4

Chord string:
*A[T44F^7 Ab7|Db^7 E7|A^7 C7|C-7 F7|Bb^7 Db7|Gb^7 A7|D-7 G7|G-7 C7 ]*A[F^7 Ab7|Db^7 E7|A^7 C7|C-7 F7|Bb^7 Ab7|Db^7 E7|A^7 C7|F^7 ]*B[C-7 F7|E-7 A7|D^7 F7|Bb^7 |Eb-7 |Ab7 |Db^7 |G-7 C7 ]*A[F^7 Ab7|Db^7 E7|A^7 C7|C-7 F7|Bb^7 Ab7|Db^7 E7|A^7 C7|F^7

Flattened measures:
| F^7Ab7      | Db^7E7      | A^7C7       | C-7F7       |
| Bb^7Db7     | Gb^7A7      | D-7G7       | G-7C7       |
| F^7Ab7      | Db^7E7      | A^7C7       | C-7F7       |
| Bb^7Ab7     | Db^7E7      | A^7C7       | F^7         |
| C-7F7       | E-7A7       | D^7F7       | Bb^7        |
| Eb-7        | Ab7         | Db^7        | G-7C7       |
| F^7Ab7      | Db^7E7      | A^7C7       | C-7F7       |
| Bb^7Ab7     | Db^7E7      | A^7C7       | F^7         |
```

atributes:

- title: The title
- composer: The composer
- style: The style (e.g. 'Swing', 'Bossa', 'Blues' etc.)
- key: The key (e.g. 'A', 'F#' etc)
- transpose: How many semitones to transpose
- comp_style: Accompaniment style (usually empty)
- bpm: Tempo in BPM (usually empty)
- repeats: How many repeats (usually empty)
- time_signature: Time signature as a tuple (e.g. (3,4), (4, 4), (5, 8) etc.)

Example attributes:

```
title:  26-2
composer:  Coltrane John
style:  Medium Up Swing
key:  F
transpose:  None
comp_style:  0
bpm:  0
repeats:  None
time_signature:  (4, 4)
```

pandas dataframe

|      |       title       | composer        |      style      | key  | transpose | repeats | time_signature |
| ---: | :---------------: | --------------- | :-------------: | :--: | :-------: | :-----: | :------------: |
|    0 |       26-2        | Coltrane John   | Medium Up Swing |  F   |   None    |  None   |     (4, 4)     |
|    1 |  500 Miles High   | Corea Chick     |   Bossa Nova    |  E-  |   None    |  None   |     (4, 4)     |
|    2 |     502 Blues     | Rowles Jimmy    |      Waltz      |  A-  |   None    |  None   |     (3, 4)     |
|    3 | 52nd Street Theme | Monk Thelonious | Up Tempo Swing  |  C   |   None    |  None   |     (4, 4)     |
|    4 |   9.20 Special    | Warren Earl     |  Medium Swing   |  C   |   None    |  None   |     (4, 4)     |

[IRealPro File Format](https://irealpro.com/ireal-pro-file-format/)

```
Chords
Chord symbol format: Root + an optional chord quality + an optional inversion

For example just a root:
C
or a root plus a chord quality
C-7
or a root plus in inversion inversion
C/E
or a root plus a quality plus an inversion
C-7/Bb

All valid roots and inversions:
C, C#, Db, D, D#, Eb, E, F, F#, Gb, G, G#, Ab, A, A#, Bb, B

All valid qualities:
5, 2, add9, +, o, h, sus, ^, -, ^7, -7, 7, 7sus, h7, o7, ^9, ^13, 6, 69, ^7#11, ^9#11, ^7#5, -6, -69, -^7, -^9, -9, -11, -7b5, h9, -b6, -#5, 9, 7b9, 7#9, 7#11, 7b5, 7#5, 9#11, 9b5, 9#5, 7b13, 7#9#5, 7#9b5, 7#9#11, 7b9#11, 7b9b5, 7b9#5, 7b9#9, 7b9b13, 7alt, 13, 13#11, 13b9, 13#9, 7b9sus, 7susadd3, 9sus, 13sus, 7b13sus, 11

Alternate Chords:
iReal Pro can also display smaller “alternate” chords above the regular chords. All the same rules apply for the format of the chord and to mark them as alternate chords you enclose them in round parenthesis:
(Db^7/F)

No Chord: n
Adds a N.C. symbol in the chart which makes the player skip harmony and bass for that measure or beats

Repeat Symbols:
x This is the “Repeat one measure” % symbol and is usually inserted in the middle of an empty measure:
r This is the “Repeat the previous two measures” symbol and is usually inserted across two empty measures:

Example: [T44C | x | x | x |D-7 |G7 | r| Z


Slash:
Sometimes we might want to add slash symbol to indicate that we want to repeat the preceding chord:
|C7ppF7|

Chord size:
When trying to squeeze many chords in one measure you might want to make them narrower.
To do this insert an s in the chord progression and all the following chord symbols will be narrower until a l symbol is encountered that restores the normal size.

Dividers:
One or more space characters are usually used to separate chords but sometimes we want to pack many chords in one measure in which case we use the comma , to separate the chords without adding empty cells to the chord progression.

Example of the chord progression of a complete song:
{*AT44D- D-/C |Bh7, Bb7(A7b9) |D-/A G-7 |D-/F sEh,A7,|Y|lD- D-/C |Bh7, Bb7(A7b9) |D-/A G-7 |N1D-/F sEh,A7} Y|N2sD-,G-,lD- ][*BC-7 F7 |Bb^7 |C-7 F7 |Bb^7 n ||C-7 F7 |Bb^7 |B-7 E7 |A7,p,p,p,][*AD- D-/C |Bh7, Bb7(A7b9) |D-/A G-7 |D-/F sEh,A7,||lD- D-/C |Bh7, Bb7(A7b9) |D-/A G-7 |D-/F sEh,A7Z

The same song’s full custom url:
irealbook://A Walkin Thing=Carter Benny=Medium Swing=D-=n={*AT44D- D-/C |Bh7, Bb7(A7b9) |D-/A G-7 |D-/F sEh,A7,|Y|lD- D-/C |Bh7, Bb7(A7b9) |D-/A G-7 |N1D-/F sEh,A7} Y|N2sD-,G-,lD- ][*BC-7 F7 |Bb^7 |C-7 F7 |Bb^7 n ||C-7 F7 |Bb^7 |B-7 E7 |A7,p,p,p,][*AD- D-/C |Bh7, Bb7(A7b9) |D-/A G-7 |D-/F sEh,A7,||lD- D-/C |Bh7, Bb7(A7b9) |D-/A G-7 |D-/F sEh,A7Z

The same song’s full custom url embedded in an HTML link:

<a href="irealbook://A Walkin Thing=Carter Benny=Medium Swing=D-=n={*AT44D- D-/C |Bh7, Bb7(A7b9) |D-/A G-7 |D-/F sEh,A7,|Y|lD- D-/C |Bh7, Bb7(A7b9) |D-/A G-7 |N1D-/F sEh,A7} Y|N2sD-,G-,lD- ][*BC-7 F7 |Bb^7 |C-7 F7 |Bb^7 n ||C-7 F7 |Bb^7 |B-7 E7 |A7,p,p,p,][*AD- D-/C |Bh7, Bb7(A7b9) |D-/A G-7 |D-/F sEh,A7,||lD- D-/C |Bh7, Bb7(A7b9) |D-/A G-7 |D-/F sEh,A7Z">A Walkin Thing</a>
```

Trzymam sobie akordy w liście słowników:

```
[{'Root': 'F', 'Ext': '^7', 'Bass': ''},
 {'Root': 'Ab', 'Ext': '7', 'Bass': ''},
 {'Root': 'Db', 'Ext': '^7', 'Bass': ''},
 {'Root': 'E', 'Ext': '7', 'Bass': ''},
 {'Root': 'A', 'Ext': '^7', 'Bass': ''},
 {'Root': 'C', 'Ext': '7', 'Bass': ''},
 {'Root': 'C', 'Ext': '-7', 'Bass': ''},
 {'Root': 'F', 'Ext': '7', 'Bass': ''},
 {'Root': 'Bb', 'Ext': '^7', 'Bass': ''},
 {'Root': 'Db', 'Ext': '7', 'Bass': ''},
 {'Root': 'Gb', 'Ext': '^7', 'Bass': ''},
 {'Root': 'A', 'Ext': '7', 'Bass': ''},
 {'Root': 'D', 'Ext': '-7', 'Bass': ''},
 {'Root': 'G', 'Ext': '7', 'Bass': ''},
 {'Root': 'G', 'Ext': '-7', 'Bass': ''},
 {'Root': 'C', 'Ext': '7', 'Bass': ''},
 {'Root': 'F', 'Ext': '^7', 'Bass': ''},
 {'Root': 'Ab', 'Ext': '7', 'Bass': ''},
 {'Root': 'Db', 'Ext': '^7', 'Bass': ''},
```



Aby dodać do data frame'a dzielę sobie akordy na coś takiego i  przetrzymuję jako string - mam metode co z tego z powrotem mi zrobi listę słowników

```
'["F", "^7", ""];["Ab", "7", ""];["Db", "^7", ""];["E", "7", ""];["A", "^7", ""];["C", "7", ""];["C", "-7", ""];["F", "7", ""];["Bb", "^7", ""];["Db", "7", ""];["Gb", "^7", ""];["A", "7", ""];["D", "-7", ""];["G", "7", ""];["G", "-7", ""];["C", "7", ""];["F", "^7", ""];["Ab", "7", ""];["Db", "^7", ""];["E", "7", ""];["A", "^7", ""];["C", "7", ""];["C", "-7", ""];["F", "7", ""];["Bb", "^7", ""];["Ab", "7", ""];["Db", "^7", ""];["E", "7", ""];["A", "^7", ""];["C", "7", ""];["F", "^7", ""];["C", "-7", ""];["F", "7", ""];["E", "-7", ""];["A", "7", ""];["D", "^7", ""];["F", "7", ""];["Bb", "^7", ""];["Eb", "-7", ""];["Ab", "7", ""];["Db", "^7", ""];["G", "-7", ""];["C", "7", ""];["F", "^7", ""];["Ab", "7", ""];["Db", "^7", ""];["E", "7", ""];["A", "^7", ""];["C", "7", ""];["C", "-7", ""];["F", "7", ""];["Bb", "^7", ""];["Ab", "7", ""];["Db", "^7", ""];["E", "7", ""];["A", "^7", ""];["C", "7", ""];["F", "^7", ""]'
```

Dobra jednak nie jednak z akordów robie od razu numery nut:

```
Bb7	Eb7	Bb7	Bb7
Eb7	Eb7	Bb7	D-7	G7
G7	C-7	F7	Bb7	G7	C-7	F7
```

```python
[[10, 14, 17, 20],
 [3, 7, 10, 13],
 [10, 14, 17, 20],
 [10, 14, 17, 20],
 [3, 7, 10, 13],
 [3, 7, 10, 13],
 [10, 14, 17, 20],
 [2, 5, 9, 12],
 [7, 11, 14, 17],
 [0, 3, 7, 10],
 [5, 9, 12, 15],
 [10, 14, 17, 20],
 [7, 11, 14, 17],
 [0, 3, 7, 10],
 [5, 9, 12, 15]]
```

Znalazłem ponad 2k piosenek.

```python
len(Counter(df["title"]).most_common())
> 2137
```

```
repetitions:  187
total:  2332
unique:  2137
```
Po usunięciu duplikatów
```
repetitions:  0
total:  2137
unique:  2137
```

### Statystyki:

```python
chords_lans = [len(ast.literal_eval(chords_str)) for chords_str in pd.read_csv("songs_and_chords.csv")["encoded_chords"]]
sum(chords_lans)/len(chords_lans)
>48.50818905007019
```

średnio 48.5 akordu na utwór.

Chords Progression Lengths Distribution:

![Chords Progression Lengths Distribution](https://github.com/SaxMan96/Harmony-Analyzer/blob/master/images/Chords%20Progression%20Lengths%20Distribution.png?raw=true)

Pierwszy eksperyment z przewidywaniem ósmego akordu:

```
Model: "sequential_7"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lstm_6 (LSTM)                (None, 32)                5760      
_________________________________________________________________
dense_6 (Dense)              (None, 12)                396       
=================================================================
Total params: 6,156
Trainable params: 6,156
Non-trainable params: 0
_________________________________________________________________
None
Accuracy: 72.68%
```

Thresholduje wyjście:

```python
pred = loaded_model.predict(X_test[0].reshape(1,7,12))[0]
print(pred)
pred = threshold_prediction(pred, 4)
print(pred)
```

```
[0.4242173  0.4184368  0.23916304 0.45015854 0.13999394 0.58093506
 0.1970064  0.22598988 0.5192512  0.14264661 0.38363022 0.17546797]
[1 0 0 1 0 1 0 0 1 0 0 0]
```

Decodowanie:

```python
print(X_train[0][0])
print(decode(X_train[0][0]))
```

```
[0 0 1 0 0 1 0 1 0 0 0 1]
[2, 5, 7, 11]
```

Pierwsze wyniki predictowania:

```
A-maj7/C  | A#m7/C#  | A-maj7/C  | Fm7/C
Fm/C  | Fm/C  | C7  | Fm/CaddB-
Gm/D  | B-7/D  | Fm/C  | Gm7/D
E-  | A-7/C  | A-7/C  | A-/CaddA
Gm7/D  | C7  | F/C  | F/CaddB-
E-  | Gm/D  | Cm  | Cm7
C7  | F-+/CaddB-  | F/C  | B-/C
B-/D  | E-+  | C7  | Chord Symbol Cannot Be Identified
F7/C  | B-/D  | B-/D  | B-/C
E-m  | A#m7/C#  | E-m  | Chord Symbol Cannot Be Identified
Gm7/D  | C7  | F/C  | B-/C
D/o7/C  | E-+addF  | Cm7  | B-/C
F9/CaddB  | B-7/D  | Cm7addD-,A-,F  | D-maj7/C
A7/C#  | D7/C  | D7/C  | Chord Symbol Cannot Be Identified
Fm7/C  | B-7/D  | E-maj7/D  | D/o7/C
Am7/C  | E7/D  | E7/D  | CaddD
B-7/D  | E-maj7/D  | B-/D  | Cm7
B-/D  | A7/C#  | B-/D  | E-addF
Gm7/D  | Gm7/D  | Cm7  | B-/C
A7/C#  | Dm  | G7/D  | Chord Symbol Cannot Be Identified
```