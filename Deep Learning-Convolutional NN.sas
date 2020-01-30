/*****************************************************************************/
/* Loading the CAS library */
/********************************************************************************************************************************/
cas; 
caslib _all_ assign;

/* setting mycas as public caslib where data has been promoted already*/
libname mycas cas caslib=public;


/*****************************/
/* Summarize the image data	 */
/*****************************/
proc cas;
image.summarizeimages / table={caslib = 'public' name='smallImageDatashuffled', where='_PartInd_=1'};
run;


/* Added code below to load Image and DeepLearn action set */
proc cas;
loadactionset 'DeepLearn';
loadactionset 'image';
run;



Proc Cas;
/*****************************/
/* Build a model shell		 */
/*****************************/
BuildModel / modeltable={name='MYCNN', replace=1} type = 'CNN';


/*****************************/
/* Add an input layer		 */
/*****************************/
AddLayer / model='MYCNN' name='data' layer={type='input' nchannels=3 width=32 height=32 std='std' offsets={114.766516,123.724943,126.078225}}; 


/*****************************/
/* b. Add a convolutional layer just after the input layer. The convolutional layer should have the following attributes:
      32 filters; Width of 3; Height of 3; Stride of 2 */
/*****************************/
AddLayer / model='MYCNN' name='ConVLayer1' layer={type='CONVO' nFilters=32  width=3 height=3 stride=2} srcLayers={'data'};


/*****************************/
/* c. Add two pooling layers and connect the convolutional layer to each of the two new pooling layers. 
      Ensure that each pooling layer has the following attributes: Width of 2; Height of 2; Stride of 1 
      Set one of the pooling layers to perform a maximum summary, and one to perform an average summary */
/*****************************/
AddLayer / model='MYCNN' name='PoolLayerMax' layer={type='POOL'  width=2 height=2 stride=1 pool='max'} srcLayers={'ConVLayer1'}; 
AddLayer / model='MYCNN' name='PoolLayerAvg' layer={type='POOL'  width=2 height=2 stride=1 pool='mean'} srcLayers={'ConVLayer1'}; 


/*****************************/
/* d. Add a concatenation layer to join the two pooling layers */
/*****************************/
AddLayer / model='MYCNN' name='concatlayer1' layer={type='concat'} srcLayers={'PoolLayerMax','PoolLayerAvg'}; 


/*****************************/
/* e. Add a convolutional layer after the concatenation layer. Structure the convolutional layer to have the following attributes:
      128 filters; Width of 3; Height of 3; Stride of 1; Use a Xavier weight initialization */
/*****************************/
AddLayer / model='MYCNN' name='ConVLayer2' layer={type='CONVO' nFilters=128  width=3 height=3 stride=1 init='xavier'} srcLayers={'concatlayer1'};


/*****************************/
/* f.Add a fully connected layer and connect the convolutional layer to the fully connected layer. 
     Connect the fully connected layer to the output layer provided in the program template. 
     The fully connected layer should have the following attributes:
     20 neurons; Use a Xavier weight initialization; Apply batch normalization; Apply an exponential linear activation transformation
     Note: You need to use two add Layer actions to complete task f */
/*****************************/
AddLayer / model='MYCNN' name='FCLayer1' layer={type='FULLCONNECT' n=20 act='Identity' init='xavier' includeBias=False} srcLayers={'ConVLayer2'};  
AddLayer / model='MYCNN' name='BatchLayer1' layer={type='BATCHNORM' act='ELU'} srcLayers={'FCLayer1'};


/*****************************/
/* Add an output layer		 */
/*****************************/
AddLayer / model='MYCNN' name='outlayer' layer={type='output' act='SOFTMAX'} srcLayers={'BatchLayer1'};
run;



/*****************************/
/* l. Modify the model by adding a dropout rate of 50% to the fully connected layer -> change the code in (f) to the below: */
/*****************************/
AddLayer / model='MYCNN' name='FCLayer1' layer={type='FULLCONNECT' n=20 act='Identity' init='xavier' dropout=.5 includeBias=False} srcLayers={'ConVLayer2'};  
AddLayer / model='MYCNN' name='BatchLayer1' layer={type='BATCHNORM' act='ELU'} srcLayers={'FCLayer1'};

AddLayer / model='MYCNN' name='outlayer' layer={type='output' act='SOFTMAX'} srcLayers={'BatchLayer1'};  /*output layer*/
run;



/*****************************/
/* p. Apply batch normalization to the convolution layer containing 128 filters 
     (remember to set the activation function to identity and remove the bias from the convolution layer) 
      -> change the code in (e) and (l) to below: */
/*****************************/
AddLayer / model='MYCNN' name='ConVLayer2' layer={type='CONVO' nFilters=128  width=3 height=3 stride=1 act='Identity' init='xavier' includeBias=False} srcLayers={'concatlayer1'};
AddLayer / model='MYCNN' name='BatchLayer2' layer={type='BATCHNORM' act='ELU'} srcLayers={'ConVLayer2'};

AddLayer / model='MYCNN' name='FCLayer2' layer={type='FULLCONNECT' n=20 act='Identity' init='xavier' dropout=.5 includeBias=False} srcLayers={'BatchLayer2'};  
AddLayer / model='MYCNN' name='BatchLayer3' layer={type='BATCHNORM' act='ELU'} srcLayers={'FCLayer2'};

AddLayer / model='MYCNN' name='outlayer' layer={type='output' act='SOFTMAX'} srcLayers={'BatchLayer3'};   /*output layer*/
run;


/*****************************/
/* t. Add a dropout rate of 10% to the convolution layer containing 128 filters -> change the code in (p) to below: */
/*****************************/
AddLayer / model='MYCNN' name='ConVLayer2' layer={type='CONVO' nFilters=128  width=3 height=3 stride=1 act='Identity' init='xavier' dropout=.1 includeBias=False} srcLayers={'concatlayer1'};
AddLayer / model='MYCNN' name='BatchLayer2' layer={type='BATCHNORM' act='ELU'} srcLayers={'ConVLayer2'};

AddLayer / model='MYCNN' name='FCLayer2' layer={type='FULLCONNECT' n=20 act='Identity' init='xavier' dropout=.5 includeBias=False} srcLayers={'BatchLayer2'};  
AddLayer / model='MYCNN' name='BatchLayer3' layer={type='BATCHNORM' act='ELU'} srcLayers={'FCLayer2'};

AddLayer / model='MYCNN' name='outlayer' layer={type='output' act='SOFTMAX'} srcLayers={'BatchLayer3'};   /*output layer*/
run;



/****************************************/
/* Train the CNN model, CoVNet			*/
/****************************************/

proc cas;
	dlTrain / table={caslib = 'public' name='SmallImageDatashuffled', where='_PartInd_=1'} model='MYCNN' 
        modelWeights={name='ConVTrainedWeights_d', replace=1}
        bestweights={name='ConVbestweights', replace=1}
        inputs='_image_' 
        target='_label_' nominal={'_label_'}
         ValidTable={caslib = 'public' name='SmallImageDatashuffled', where='_PartInd_=2'} 
 
        optimizer={minibatchsize=60, 
        			algorithm={method='ADAM', lrpolicy='Step', gamma=0.5,
       							beta1=0.9, beta2=0.999, learningrate=.001}
        			maxepochs=80} 
        seed=12345
;
run;
