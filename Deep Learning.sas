/*****************************************************************************/
/*  Create a default CAS session and create SAS librefs for existing caslibs */
/*  so that they are visible in the SAS Studio Libraries tree.               */
/*****************************************************************************/

cas; 
caslib _all_ assign;

/* setting mycas as public caslib where data has been promoted already*/

libname mycas cas caslib=public;


/****************************************/
/* Uncomment and run the below three    */ 
/* lines of code if the DeepLearn       */ 
/* action set has not been loaded.		*/
/****************************************/
/* proc cas; */
/* loadactionset 'DeepLearn'; */
/* run; */


/* WithBatch Normalization */
proc cas;
BuildModel / modeltable={name='BatchDLNN', replace=1} type = 'DNN';

/* 			INPUT Layer 				*/
AddLayer / model='BatchDLNN' name='data' layer={type='input' STD='STD' dropout=.05}; 

/* ADD A HIDDEN LAYER RIGHT AFTER THE INPUT LAYER (connect this hidden layer to both the input layer and the next hidden layer	*/
AddLayer / model='BatchDLNN' name='HLayer0' layer={type='FULLCONNECT' n=40 dropout=.05 act='identity' init='xavier' includeBias=False} srcLayers={'data'};
AddLayer / model='BatchDLNN' name='BatchLayer0' layer={type='BATCHNORM' act='ELU'} srcLayers={'HLayer0'};

/* 			FIRST HIDDEN LAYER 			*/
AddLayer / model='BatchDLNN' name='HLayer1' layer={type='FULLCONNECT' n=30 act='ELU' init='xavier' } srcLayers={'BatchLayer0'};

/* 			SECOND HIDDEN LAYER 		*/
AddLayer / model='BatchDLNN' name='HLayer2' layer={type='FULLCONNECT' n=20 act='identity' init='xavier' includeBias=False} srcLayers={'HLayer1'};
AddLayer / model='BatchDLNN' name='BatchLayer2' layer={type='BATCHNORM' act='ELU'} srcLayers={'HLayer2'};

/* 			THIRD HIDDEN LAYER 			*/
AddLayer / model='BatchDLNN' name='HLayer3' layer={type='FULLCONNECT' n=10 act='identity' init='xavier' includeBias=False } srcLayers={'BatchLayer2'};  
AddLayer / model='BatchDLNN' name='BatchLayer3' layer={type='BATCHNORM' act='ELU'} srcLayers={'HLayer3'};

/* 			FOURTH HIDDEN LAYER 		*/
AddLayer / model='BatchDLNN' name='HLayer4' layer={type='FULLCONNECT' n=5 act='identity' init='xavier' includeBias=False } srcLayers={'BatchLayer3'};  
AddLayer / model='BatchDLNN' name='BatchLayer4' layer={type='BATCHNORM' act='ELU'} srcLayers={'HLayer4'};

/* 			FIFTH HIDDEN LAYER 			*/
AddLayer / model='BatchDLNN' name='HLayer5' layer={type='FULLCONNECT' n=10 act='identity' init='xavier' includeBias=False } srcLayers={'BatchLayer4'};  
AddLayer / model='BatchDLNN' name='BatchLayer5' layer={type='BATCHNORM' act='ELU'} srcLayers={'HLayer5'};

/* 			SIXTH HIDDEN LAYER 			*/
AddLayer / model='BatchDLNN' name='HLayer6' layer={type='FULLCONNECT' n=20 act='identity' init='xavier' includeBias=False} srcLayers={'BatchLayer5'};
AddLayer / model='BatchDLNN' name="BatchLayer6" layer={type='BATCHNORM' act='ELU'} srcLayers={'HLayer6'};

/* 			SEVENTH HIDDEN LAYER 		*/
AddLayer / model='BatchDLNN' name='HLayer7' layer={type='FULLCONNECT' n=30 act='identity' init='xavier' includeBias=False } srcLayers={'BatchLayer6'};     
AddLayer / model='BatchDLNN' name="BatchLayer7" layer={type='BATCHNORM' act='ELU'} srcLayers={'HLayer7'};


/* ADD A HIDDEN LAYER JUST BEFORE THE OUTPUT LAYER (connect this hidden layer to both the previous hidden layer and the output layer */
AddLayer / model='BatchDLNN' name='HLayer8' layer={type='FULLCONNECT' n=40 act='ELU' init='xavier' includeBias=False} srcLayers={'BatchLayer7'};

/*          OUTPUT LAYER                  */
AddLayer / model='BatchDLNN' name='outlayer' layer={type='output' act='LOGISTIC'} srcLayers={'HLayer8'};
run;



proc cas;
dlTrain /table={caslib='public' name = 'Train_Develop'} model='BatchDLNN' 
        modelWeights={name='BatchTrainedWeights_d', replace=1}
        bestweights={name='bestbatchweights', replace=1}
        inputs={'AcctAge',
        		'DDABal',
        		'CashBk',
        		'Checks',
        		'NSFAmt',
        		'Phone',
        		'Teller',
        		'SavBal',
        		'ATMAmt',
        		'POS',
        		'POSAmt',
        		'CDBal',
        		'IRABal',
        		'LOCBal',
        		'ILSBal',
        		'MMBal',
        		'MMCred',
        		'MTGBal',
        		'CCBal',
        		'CCPurc',
        		'Income',
        		'LORes',
        		'HMVal',
        		'Age',
        		'CRScore',
        		'Dep',
        		'DepAmt',
        		'InvBal',
        		'DDA',
        		'DirDep',
        		'NSF',
        		'Sav',
        		'ATM',
        		'CD',
        		'IRA',
        		'LOC',
        		'ILS',
        		'MM',
        		'MTG',
        		'CC',
        		'SDB',
        		'HMOwn',
        		'Moved',
        		'InArea',
        		'Inv' 
        		}
      nominals={'INS',
      			'DDA',
        		'DirDep',
        		'NSF',
        		'Sav',
        		'ATM',
        		'CD',
        		'IRA',
        		'LOC',
        		'ILS',
        		'MM',
        		'MTG',
        		'CC',
        		'SDB',
        		'HMOwn',
        		'Moved',
        		'InArea', 
				'Inv'
        		 }
	ValidTable={caslib='public' name = 'Valid_Develop'}
	target='INS'  
         optimizer={minibatchsize=60, algorithm={method='ADAM', lrpolicy='Step', gamma=0.5, stepsize=10,
       				beta1=0.9, beta2=0.999, learningrate=.001} 
       	regL1=0.003, 
       	regL2=0.002,  
       	maxepochs=100} 
       	seed=12345
;
run;

