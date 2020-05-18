# Model

Model LSTM architecture - build using `Keras` from `TensorFlow 2.0`

![LTSTM_30_model](F:\MiNI IAD\_Master_Thesis\Project\images\model\LTSTM_30_model.png)`
# Data

I've used an embedded data. I've created  a samples of sequences of size defined by `sample_size` e.g. 4. The last chord was used as an predicted value and first 3 (`sample_size = 4`) were used to feed LSTM. 

I've spitted data on three sets:  

- `60%` Train data `~750k` examples
- `20%` Test data  `~250k` examples
- `20%` Valid data  `~250k` examples

Data distribution is close to standard and I didn't performed additional standardization.


# Metric

I've used a custom metric to observe the process of training on my embedded data - `mean_cosine_distance`.

`cosine_distance`is equal to `1-cosine_similarity` which is a cosine of an angle between two multidimensional vectors.  I've added it as a `Callback` to in method `fit()` in Keras API.

# Parameters

- `Huber`loss function due to it's known to it's smaller sensitivity to outliers than MSE - loss function.

- `batch_size` set to `5000`  -  that gives around 150  batches on train set.

- optimizer `Adam(lr=1e-3)` - standard approach

# Training

30 epochs of training gave me following results.

<table>
  <tr>
    <td><img src="F:\MiNI IAD\_Master_Thesis\Project\images\model\LTSTM_30_loss.png" alt="LTSTM_30_loss" style="zoom:67%;" />
	</td>
	<td><img src="F:\MiNI IAD\_Master_Thesis\Project\images\model\LTSTM_30_mean_cos_dist.png" alt="LTSTM_30_mean_cos_dist" style="zoom:67%;" />
	</td>
  </tr>
</table>

# Evaluation

Model started slightly overfitting 10 epoch. Training may result in better result if continued, but overfitting might be a case. The `mean_cos_dist` plot shows results of training using mu custom metric, as me can see an angle between prediction and actual embedded vector is decreasing which is of course a good sign.

# Examples

Below I've presented a result of model prediction. In the last column I'm showing a list of top 5 most similar chords to output of the network. By most similar I mean a chords that have embedded representation that have the biggest cosine similarity.

```
| 	input to network   	|	true chord 	|	prediction (most similar chords)	
    Dbo7	G^7		E7			( B-7 )				['Ebh', 'Ebh9', 'A7', 'Ebh7', 'A6']
    B6		Gb6		Gb6		  ( B )					['Db-11', 'Db2', 'A6', 'Db-#5', 'Dbh9']
    D7		Db7		Gb-			( Eb7sus )		['Db-b6', 'Ebh', 'A6', 'Ebh7', 'A^7']
    Bb7		Eb		Eb7			( Db7 )				['A6', 'E+', 'Co', 'Gb+', 'A']
    Ab^7	Db^7	Ab^7		( B7 )				['Ebh', 'A6', 'Db-', 'Ebh7', 'Co7']
    Gb^7	Bbh7	Eb7b9		( F7 )				['Ebh', 'Db-b6', 'Ebh7', 'Gb-b6', 'Db-']
    A		Db7		Db7		( C6 )		['Ebh9', 'Ebh', 'Db-6', 'Db-', 'A^7']
    Eb7		C7		F-7		( D7 )		['A6', 'Ebh', 'Db-', 'Gb+', 'A']
    A7		Eh		A7		( Bb^7 )	['Ebo', 'Ebo7', 'A6', 'Ebh', 'A+']
    Db-7	Gb7		B^7		( A7 )		['Ebh', 'Ebh9', 'Ebh7', 'A6', 'Eb-7']
    Bb-7	Ab-7	Bb-7	( D7 )		['Co7', 'Ebh', 'Db-', 'A^7', 'Db-b6']
    C7		F-7		Bb7		( E-7 )		['A6', 'Ebh', 'B-b6', 'Db-', 'Db-b6']
    Eb		Bb6		Eh7		( G7 )		['Ebh', 'A6', 'Db-', 'Ebh7', 'B-7']
    G^7		E7		Gb-7	( A^7 )		['A+', 'Ebh', 'A6', 'A', 'Gb-6']
    C7		B^7		D7		( G )		['Ebh', 'A6', 'Ebh9', 'Ebh7', 'Do7']
    E7		A		A		( D-7 )		['Ebh9', 'Ebh', 'Ebh7', 'Db-6', 'Eb7b9#9']
    Eb7		Ab7		Db		( C-7 )		['A^', 'Db-', 'A6', 'A', 'Db-b6']
    D7#5	D7#5	G-11	( D7b9 )	['A^7', 'Co7', 'C^7', 'Ebh7',  Gb7b5']
    Eb-		Ab7		Ab7		( Bb7 )		['Ebh', 'A^', 'A6', 'Db-b6', 'Ebh9']
    Eb-7	Ab7		Db^7	( G )		['Ebh', 'A^7', 'Db-', 'A6', 'A^']
```

Another example of song generated from initial sequence 

```
IN: ( A^7		|	F#7 |	B-7	|	E7	|	) 
 			A6		|	E7	|	A6	|	G7	|	  
OUT: 	G7		|	A^7	|	E7	|	A^7	|	  
  		Eb-7b5|	E7	|	A6	|	A^7	|	  
```