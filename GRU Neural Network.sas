cas; 
caslib _all_ assign;
libname mycas cas caslib=public;

/*a. Print the first few observations of MOVIE_CLEAN and MOVIE_EMBED to view the data sets*/
Title "First 10 obs of MOVIE_CLEAN DATA SET";
Proc print data = Public.MOVIE_CLEAN (obs=10);
run;

Title "First 10 obs of MOVIE_EMBED DATA SET";
Proc print data = Public.MOVIE_EMBED (obs=10);
run;
title;


/*b.Use the FREQ procedure to view the number of movies that earned a profit */
Proc Freq data=Public.MOVIE_CLEAN;
Tables Profit/ nocum;
run;


/*c.Find and print the movie titles that mention Denzel Washington in the movie overview */
Proc SQL;
Select * from Public.MOVIE_CLEAN
Where propcase(Overview) contains "Denzel Washington"
Order by Title;
quit; 

/*Alternative code: 
    data public.DW (drop=newvar);
	set Public.MOVIE_CLEAN;
	newvar = find(overview,'denzel washington','i');
	if newvar>0;
    run;

    proc print data=public.DW;
    run;  
*/


/*d.Partition the data into 70% training, 15% validation, and 15% for testing by adding a partition indicator to the CAS table. Use Seed 802 for data partitioning*/
proc partition data=Public.MOVIE_CLEAN
	samppct=70 samppct2=15 seed=802 partind;
	output out=Public.MOVIE_CLEAN_2;
run;

proc freq data=Public.MOVIE_CLEAN_2;
	tables profit _partind_ profit*_partind_;
run;


/*e.Use the shuffle action from the table action set to randomize the observations and avoid a potential ordering bias in the deep learning model*/
proc cas;
	table.shuffle / 
	table = {caslib='public',name = 'MOVIE_CLEAN_2'}
	casout = {caslib = 'public', name='MOVIE_CLEAN_2', replace=True};
quit;


/*f.   Use the deepLearn action set to build a GRU neural network with one input layer, two GRU hidden layers, and an output layer.
1)   Use the buildModel action to initialize the RNN and then add an input layer.
2)   Connect the input layer to a GRU hidden layer with 15 neurons, set the activation function to auto, set initialization to Xavier, and set the output type to same length.
3)   Connect this hidden layer to another GRU hidden layer with the same arguments except set output type to encoding.
4)   Finally, connect the second hidden layer to the output layer and set the error function to auto.
5)   View the model to make sure the structure is correct using the modelInfo action. */

proc cas;
	loadactionset "deeplearn";
quit;

proc cas;
	deepLearn.buildModel / 
    model = {name='rnn',replace=True}
    type = 'RNN';

	deepLearn.addLayer /
    model = 'rnn'
    layer = {type='input'}
    replace=True
    name = 'data';

	deepLearn.addLayer / 
    model = 'rnn'
    layer = {type='recurrent', n=15, act='auto', init='xavier', rnnType='gru', reverse=True, outputType='samelength'}
    srcLayers = 'data'
    replace=True
    name = 'rnn1';

	deepLearn.addLayer / 
    model = 'rnn'
    layer = {type='recurrent', n=15, act='auto', init='xavier', rnnType='gru', reverse=True, outputType='encoding'}
    srcLayers = 'rnn1'
    replace=True
    name = 'rnn2';

	deepLearn.addLayer / 
    model = 'rnn'
    layer = {type='output', act='auto', init='xavier', error='auto'}
    srcLayers = 'rnn2'
    replace=True
    name = 'output';

	deepLearn.modelInfo / 
    model='rnn';
quit;


/* g. Use the dlTrain action to train the GRU model using the Profit variable as the target and the Overview variable as the input. 
      Train the model using the Adam optimization algorithm and a learning rate of 0.05. Use mini batch sizes of 50 and train for 20 epochs. 
      Be sure to save the weights to score the test data after the model is built. Comment on the training output. 
      Note: Use all other Optimization parameters as in Demo DLUS03D01 RNN model.					*/

proc cas;
	deepLearn.dlTrain /
    table    = {caslib = 'public',name = 'MOVIE_CLEAN_2', where = '_PartInd_ = 1'}
    validTable = {caslib = 'public',name = 'MOVIE_CLEAN_2', where = '_PartInd_ = 2'}
    target = 'profit'
    inputs = 'overview'
    texts = 'overview'
    textParms = {initInputEmbeddings={caslib = 'public',name='MOVIE_EMBED'}}
    nominals = 'profit'
    seed = '649'
    modelTable = 'rnn'
    modelWeights = {name='rnn_trained_weights', replace=True}
    optimizer = {miniBatchSize=50, maxEpochs=20, 
                     algorithm={method='adam', beta1=0.9, beta2=0.999, 
                                learningRate=0.05, gamma=0.5, lrpolicy='step', stepsize=15, 
								clipGradMax=10, clipGradMin=-10}};
quit;


/*h.Score the test data and view the misclassification error. Comment	*/

proc cas;
	deepLearn.dlScore / 
    table    = {caslib = 'public',name = 'MOVIE_CLEAN_2', where = '_PartInd_ = 0'}
    model = 'rnn'
    initWeights = 'rnn_trained_weights'
    copyVars = 'profit'
    textParms = {initInputEmbeddings={caslib = 'public',name='MOVIE_CLEAN_2'}}
    casout = {name='rnn_scored', replace=True};
quit;



/*i.Regularize the previous GRU model by building the model again but include a dropout of 0.40 in each GRU hidden layer. 
    Train the new model with the same arguments for the dlTrain action and view the changes in the optimization history. Comment on what the differences are from step (g)*/

/*change Step (f) & (g) to the below code */

proc cas;
	deepLearn.buildModel / 
    model = {name='rnn_regularize',replace=True}
    type = 'RNN';

	deepLearn.addLayer /
    model = 'rnn_regularize'
    layer = {type='input'}
    replace=True
    name = 'data';

	deepLearn.addLayer / 
    model = 'rnn_regularize'
    layer = {type='recurrent', n=15, act='auto', init='xavier', dropout=0.4, rnnType='gru', reverse=True, outputType='samelength'}
    srcLayers = 'data'
    replace=True
    name = 'rnn1';

	deepLearn.addLayer / 
    model = 'rnn_regularize'
    layer = {type='recurrent', n=15, act='auto', init='xavier', dropout=0.4, rnnType='gru', reverse=True, outputType='encoding'}
    srcLayers = 'rnn1'
    replace=True
    name = 'rnn2';

	deepLearn.addLayer / 
    model = 'rnn_regularize'
    layer = {type='output', act='auto', init='xavier', error='auto'}
    srcLayers = 'rnn2'
    replace=True
    name = 'output';

	deepLearn.modelInfo / 
    model='rnn_regularize';
quit;


proc cas;
	deepLearn.dlTrain /
    table    = {caslib = 'public',name = 'MOVIE_CLEAN_2', where = '_PartInd_ = 1'}
    validTable = {caslib = 'public',name = 'MOVIE_CLEAN_2', where = '_PartInd_ = 2'}
    target = 'profit'
    inputs = 'overview'
    texts = 'overview'
    textParms = {initInputEmbeddings={caslib = 'public',name='MOVIE_EMBED'}}
    nominals = 'profit'
    seed = '649'
    modelTable = 'rnn_regularize'
    modelWeights = {name='rnn_regularized_trained_weights', replace=True}
    optimizer = {miniBatchSize=50, maxEpochs=20, 
                     algorithm={method='adam', beta1=0.9, beta2=0.999, 
                                learningRate=0.05, gamma=0.5, lrpolicy='step', stepsize=15, 
								clipGradMax=10, clipGradMin=-10}};
quit;


/*j.Score the test data using the GRU model with regularization. Comment */

proc cas;
	deepLearn.dlScore / 
    table    = {caslib = 'public',name = 'MOVIE_CLEAN_2', where = '_PartInd_ = 0'}
    model = 'rnn_regularize'
    initWeights = 'rnn_regularized_trained_weights'
    copyVars = 'profit'
    textParms = {initInputEmbeddings={caslib = 'public',name='MOVIE_CLEAN_2'}}
    casout = {name='rnn_regularized_scored', replace=True};
quit;
