
#include "neuralnetwork2.h"
//#include "cnn_both.h"
#include <sys/stat.h>
#include <algorithm>
#include <map>
#include <cstdlib>
#include <sstream>


class Stats {
public:
	static double avgDiff(double v) {
		return (pow(v,2)  - v + 0.5);
	}
	// note: this benchmarking fails to account for the distribution of expected editing frequencies
};

int main(int argc, char*argv[]) {

	int jobID = 1;

    if(argc < 6) {
		//cout << "Usage: ./stats <dataset> <filesuffix>" << endl;
		//cout << "Usage: ./stats <dataset> <filesuffix>" << endl;
		cout << "Usage: ./stats <jobfile> <dataset> <runSize> <partitionSize> <mutate?> optional(<benchmarkA> <benchmarkB>)" << endl;
		//cout << "Job file (per line): <filesuffix> <stochastic?> <numIcols> <filtersize> <convlayersize> <...layersize...> <trainings>" << endl;
		cout << "Job file              (per line): <filesuffix> <stochastic?> <filtersize?> <numIcols> <...layersize...> <trainings>" << endl;
		cout << " <stochastic?>: neuralnetwork will adjust weights each trial based off randomly chosen training data" << endl;
		cout << " <filtersize?>: if a non-zero filtersize is specified, neural network will contain one convolutional layer with numIcols[0] sectors" << endl;
		cout << "Benchmark and dataset (per line): <sequence> <score [0,1]>" << endl;
		exit(0);
	}
	bool benchmark = false;
	ifstream benchmarkA, benchmarkB;
	map<string, double> predA;
	map<string, double> predB;

	if(argc >= 8) {
		benchmark = true;
		benchmarkA.open(argv[6]);
		benchmarkB.open(argv[7]);
		if(!benchmarkA || !benchmarkB) {
			cout << "One or more benchmark datasets not good..." << endl;
			exit(1);
		}
		// read in the datasets and write them to the map
		double pred;
		string seq;
		while(true) {
			benchmarkA >> seq;
			if(!benchmarkA)
				break;
			benchmarkA >> pred;
			predA[seq] = pred;

			benchmarkB >> seq;
			benchmarkB >> pred;
			predB[seq] = pred;

		}
		cout << "Read in benchmarking predictions into hashmap A and B" << endl;
		//cout << predA["TATCAGGTTCCATAGAACCA"] << endl;
	}

	bool mutate = true;
	if(atoi(argv[5]) == 0) mutate = false;

	if(mutate) cout << "Will write new jobs based on previous best performing hyperparameters" << endl;
	// Note: all of the notes before a number (ex. 1) are done with check if mutate == true.
	// () before deciding the next set of hyperparameters, it will read the best one from the job file.
	// 1. Then, it will save all of those hyperparameters to the BR prepended variables.
	// 2. then, it will mutate the hyperparameters.
	// () finally, After Testing, it will write the hyperparameters it used to the output file, as normal.
	// 3. if the hyperparametrs did better than the best one (better Prediction Score), it will write them to the job file.
	//       (cont.) Otherwise, it will write the BR hyperparameters to the fstream job file.
	// 
	
	double bestPredictionScore = 0;

	string inputfilename = argv[2];
	int runSize = atoi(argv[3]);
	int testSize = atoi(argv[4]);

	ifstream inputfile(inputfilename.c_str());
    if(!inputfile) {
    	cerr << "Inputfilename: " << inputfilename << endl;
    	cerr << "Data file not good, terminating program." << endl;
    	exit(1);
    }

    std::vector<string> staticinputs;
	std::vector<double> staticexpecteds;
	char c;
	string input;
	double expected;

	// get all of the data
	//get all of the data
    while(true) {
    	for(int i = 0; i < 20; i++) {
    		inputfile >> c;
    		if(!inputfile || c == 'q') break;
    		input+=c;
    	}
    	if(!inputfile || c == 'q') break;
    	inputfile >> expected;

    	staticinputs.push_back(input);
    	input = "";
    	staticexpecteds.push_back(expected);
    	//output7 << expected << endl;
    }

    // generate the subsets

	// training data
	string** subsetInputs = new string*[runSize];
	double** subsetExpecteds = new double*[runSize];
	// testing data
	string** testInputs = new string*[runSize];
	double** testExpecteds = new double*[runSize];

	int trainingsize = staticinputs.size() - testSize;
	for(int i = 0; i < runSize; i++) {

		// generate one subset it will use
		int* numbers = new int[(int)staticinputs.size()];
		for(int j = 0; j < (int)staticinputs.size(); j++) {
			numbers[j] = j;
		}
		random_shuffle(numbers,numbers + staticinputs.size());
		subsetInputs[i] = new string[trainingsize];
		subsetExpecteds[i] = new double[trainingsize];
		testInputs[i] = new string[testSize];
		testExpecteds[i] = new double[testSize];

		for(int j = 0; j < (int)staticinputs.size(); j++) {

			if(j < testSize) {
				testInputs[i][j] = staticinputs[numbers[j]];
				testExpecteds[i][j] = staticexpecteds[numbers[j]];
			} else {
				subsetInputs[i][j - testSize] = staticinputs[numbers[j]];
				subsetExpecteds[i][j - testSize] = staticexpecteds[numbers[j]];
			}
		}
		delete[] (numbers);
	}

	cout << "DEBUG:: determined the training sets for all " << runSize << " different trials" << endl;

	fstream jobfile;
	jobfile.open(argv[1], ios::in | ios::out);
	if(!jobfile) {
		cerr << "Job file not good, terminating program." << endl;
		exit(1);
	}

	string jobfolder = argv[1]; jobfolder += "_results";

	mkdir( (jobfolder.c_str()), 0777);

	string psDataname = "./" + jobfolder + "/"; psDataname += argv[1]; psDataname += "_results.tsv";

	// stores all of the prediction scores for all of the runs
	ofstream psData(psDataname);
	if(!psData) {
		exit(0);
	}
	string psDataHeader = "Dataset\tFilesuffix\tRunSize\tTestsperRun\tStochastic?\t...Layersizes...\tTotalWeights\tnumTrainings\tRandomPredictionScore\tBenchmarkAPredictionScore\tBenchmarkBPredictionScore\tNotes";
	psData << psDataHeader << endl;
//exit(0);
	int jobNumber = 0;
	srand (time(NULL));
	string originalExtension;

	

while(true) {
	if(!jobfile.is_open())jobfile.open(argv[1], ios::in | ios::out);

	jobNumber ++;
	cout << "Starting job number " << jobNumber << endl;

	//string inputfilename; jobfile >> inputfilename;
	string extension; jobfile >> extension;

	if(jobNumber == 1)  originalExtension = extension;

	if(jobfile.bad()) {
		cout << "bad job file" << endl;
	}

	bool stochastic; jobfile >> stochastic;
	if(!jobfile) {
		cout << "Job file empty..." << endl;
		exit(0);
	}
	
	mkdir( ("./" + jobfolder + "/" + extension).c_str(), 0777);

	double sumofDifferencesAllRuns = 0;
	double sumofAvgDifferencesAllRuns = 0;
	double sumofRandomDifferencesAllRuns = 0;
	double sumofBenchmarkADifferencesAllRuns = 0;
	double sumofBenchmarkBDifferencesAllRuns = 0;

    string output10filename = "./" + jobfolder + "/" + extension+ "/alloutputs." + extension + ".tsv";     //NEW
    string output11filename = "./" + jobfolder + "/" + extension+ "/statsPERsubset." + extension + ".tsv"; //NEW

    string output12filename = "./" + jobfolder + "/" + extension+ "/runinfo." + extension + ".tsv"; //NEW

    ofstream output10(output10filename);
    ofstream output11(output11filename);
    ofstream output12(output12filename);

    output10 << "Actual\tNNW\tRandomData\tBenchmark_A\tBenchmark_B\tNNWDiff\tRandomDatadiff\tBenchmark_Adiff\tBenchmark_Bdiff\tRandomDatadiff-NNWdiff\tBenchmarkAdiff-NNWdiff\tBenchmarkBdiff-NNWdiff" << endl; 

    output11 << "Subset\tFinalAvgLoss\tsumofDifferences\tsumofRandomDifferences\tsumofBenchmarkADifferences\tsumofBenchmarkBDifferences" << endl;

    output12 << psDataHeader << endl;
	
	std::vector<int> internalCols {};

	int filtersize; jobfile >> filtersize;

	int numInternalCols; jobfile >> numInternalCols;

	internalCols.resize(numInternalCols);
	for(int i = 0; i < numInternalCols; i++) {
		jobfile >> internalCols[i];
		//internalCols[i] = atoi(argv[7 + i]);
	}
	

	
    //int runTimes = atoi(argv[7 + numInternalCols]);
    int runTimes; jobfile >> runTimes;
    bool silentRun = false;

    // 1. save the parameters of the best run 

    //int BRrunSize = runSize;
    //int BRtestSize = testSize;
    bool BRstochastic = stochastic;
    int BRnumInternalCols = numInternalCols;
   // int BRfiltersize = filtersize;
    std::vector<int> BRinternalCols = internalCols;
    int BRrunTimes = runTimes;

    if(mutate) {
	    // 2. mutate the data!!
	    while(true) {
	    	int mutation = rand() % 150;
	    	if(mutation < 30) {
	    		cout << "Mutation: New Column" << endl;
	    		// NEW COL
	    		internalCols.push_back( (rand() % 8) + 2 );
	    		numInternalCols ++;
	    	} else if (mutation < 60) {
	    		cout << "Mutation: Delete Column" << endl;
	    		// DEL COL
	    		if(internalCols.size() > 2) {
	    			internalCols.pop_back();
	    			numInternalCols --;
	    		}
	    	} else if(mutation < 80) {
	    		cout << "Mutation: Increase Internal Nodes" << endl;
	    		// increase internal col nodes
	    		int iCol = rand() % internalCols.size();
	    		internalCols[iCol] ++;
	    	} else if(mutation < 100){
	    		cout << "Mutation: Decrease Internal Nodes" << endl;
	    		// decrease internal col nodes
	    		int iCol = rand() % internalCols.size();
	    		if(internalCols[iCol] > 1) internalCols[iCol] --;
	    	} else if(mutation < 130) {
	    		cout << "Mutation: Toggle Stochastic" << endl;
	    		if(stochastic) stochastic = false;
	    		else stochastic = true;
	    	} else if(mutation < 140) {
	    		cout << "Mutation: Increase Amount of Trainings" << endl;
	    		runTimes += 1000;
	    	} else if(mutation < 150) {
	    		cout << "Mutation: Decrease Amount of Trainings" << endl;
	    		if(runTimes > 2000)
	    		runTimes -= 1000;
	    	}
	    	// will mutate at least once (unless misses a mutation)
	    	if(rand() % 100 < 50) break;
	    }

	}

	int totalWeights = 0;


	// creates random samples 10 times
	for(int a = 0; a < runSize ; a ++ ) {

		cout << "Iteration: " << a << endl;

		cout << "Filtersize: " << filtersize << endl;

		NeuralNetwork* nnw;
		//nnw = new NeuralNetwork(20, filtersize, internalCols, 1);
		if(filtersize == 0) {

			nnw = new NeuralNetwork(20, internalCols, 1);
		}
		else {
			nnw = new NeuralNetwork(20, filtersize, internalCols, 1);
			//cout << "Not to test" << endl;
			//nnw = nullptr;
			//exit(1);
		}
			//nnw = new Neuralnetwork(20, filtersize, internalCols, 1);

		if(a == 0) {
			cout << "Total weights: " << nnw -> howManyWeights() << endl;
			nnw->display();
		}

		totalWeights = nnw -> howManyWeights();

		//cout << "Created neural network" << endl;

		
	    if(runTimes > 1000) {
	    	silentRun = true;
	    	cout << "Runtimes > 1000, Printing results only every 1000 runs." << endl;
	    }


		cout << "training data: " << endl;
		for(int i = 0; i < trainingsize; i++) {
			cout << setw(3) << i << setw(10) << subsetInputs[a][i] << " " << subsetExpecteds[a][i] << endl;
		}
		cout << endl;
		
		// train the data

		double finalAvgLoss;
		

	    bool silent = false;
	    double losses = 0;
	    int highLoss = 0;
	    double loss = 0;
	    int randomtraining;
	    for(int i = 0; i < runTimes; i++) {
	    	if(silentRun) {
	    		if(i%1000 == 0) {
	    			silent = false;
	    		}else silent = true;
	    	}
	    	for(int j = 0; j < trainingsize; j++) {
	    		if(!silent)cout << subsetInputs[a][j] << " ";
	    		if(!silent)
	    		loss = nnw -> runInput(subsetInputs[a][j], subsetExpecteds[a][j], false);
	    		else {
	    			if(!stochastic)
	    			loss = nnw -> runInput(subsetInputs[a][j], subsetExpecteds[a][j], true);
	    			if(stochastic) {
	    				randomtraining = rand()%trainingsize;
	    				loss = nnw -> runInput(subsetInputs[a][randomtraining], subsetExpecteds[a][randomtraining], true);
	    			}
	    			
	    		}
	    		
	    	    losses += loss;
	    	    if(loss > 0.0025) highLoss ++;
	    	}
	    	if(i == runTimes - 1) finalAvgLoss = losses / trainingsize;
	    	if(!silent)cout << "   >>>>> Average Loss: " << losses / trainingsize << endl;
	    	if(!silent)cout << "   >>>>> Total High Losses (> 0.05 diff): " << highLoss << endl;
	    	if(!silent) {
	    		cout << "Training iteration number: " << i << "/" << runTimes << endl;
	    		cout << "Subset iteration " << a  << "/" << runSize << endl;
	    		cout << "Job number " << jobNumber << endl;
	    	}
	    	//if(!silent)
	    	//nnw->display();
	    	nnw -> AdjustWeights(true);
	    	
	    	losses = 0;
	    	highLoss = 0;
	    }

	    // run the partitioned data
	    double diff;
	    double avgdiff;
	    double randomdata;
	    double randomdiff;
	    double sumofDifferences = 0;
	    double sumofAvgDifferences = 0;
	    double sumofRandomDifferences = 0;

	    double sumofBenchmarkADifferences = 0;
	    double sumofBenchmarkBDifferences = 0;

	    double nnwprediction;
	    double predictionA, predictionB;
	    double difA, difB;
	    
	    
	    for(int i = 0; i < testSize; i++) {
	    	cout << testInputs[a][i] << " OUTPUT: ";
	    	nnw -> runInput(testInputs[a][i], -1, false);
	    	cout << " " << setw(10) << testExpecteds[a][i];

	    	nnwprediction = nnw -> outputs -> nodes[0] -> activation;

	    	diff = abs(nnwprediction -  testExpecteds[a][i]);
	    	//output4 << nnwprediction << ", " << testExpecteds[a][i] << endl;
	    	avgdiff = Stats::avgDiff(testExpecteds[a][i]);
	    	randomdata = testExpecteds[a][rand()%testSize];
	    	randomdiff = abs(randomdata - testExpecteds[a][i]);
	    	cout << "diff: " << setw(10) << diff;
	    	cout << "avg. diff: " << setw(10) << avgdiff;
	    	cout << "rand. diff: " << setw(10) << randomdiff;

	    	if(benchmark) {
		    	predictionA = predA[testInputs[a][i]];
		    	predictionB = predB[testInputs[a][i]];
		    	difA = abs(predictionA - testExpecteds[a][i]);
		    	difB = abs(predictionB - testExpecteds[a][i]);

		    	// output the benchmarked results
		    	cout << "A: " << setw(10) << predictionA;
		    	cout << "diff " << setw(10) << difA; 

		    	cout << "B: " << setw(10) << predictionB;
		    	cout << "diff " << setw(10) << difB;

		    	sumofBenchmarkADifferences += difA;
		    	sumofBenchmarkBDifferences += difB;

		    	// "Actual\tNNW\tRandomData\tBenchmark_A\tBenchmark_B\tNNWDiff\tRandomDatadiff\tBenchmark_Adiff\tBenchmark_Bdiff\tRandomDatadiff-NNWdiff\tBenchmarkAdiff-NNWdiff\tBenchmarkBdiff-NNWdiff"
		    	output10 << testExpecteds[a][i] << '\t' << nnwprediction << '\t' << randomdata << '\t' << predictionA << '\t' << predictionB << '\t' 
		    	  << diff << '\t' << randomdiff << '\t' << difA << '\t' << difB << '\t' << (difA - diff) << '\t' << (difB - diff) << '\t' << (randomdiff - diff) << endl;
	    	}

	    	cout << endl;

	    	//
	    	sumofDifferences += diff;
	    	sumofAvgDifferences += avgdiff;
	    	sumofRandomDifferences += randomdiff;
	    	//output << diff << endl;
	    	//output2 << avgdiff << endl;
	    	//output3 << randomdiff << endl;
	    	//delete(nnw);
	    	
	    }
		
	    cout << setw(20) << "Sum of Differences : " << sumofDifferences << endl;
	    cout << setw(20) << "Sum of avg. Differences : " << sumofAvgDifferences << endl;
	    cout << setw(20) << "Sum of random Differences : " << sumofRandomDifferences << endl;
	    cout << setw(20) << "Sum of BenchmarkA Differences : " << sumofBenchmarkADifferences << endl;
	    cout << setw(20) << "Sum of BenchmarkB Differences : " << sumofBenchmarkBDifferences << endl;
	    if(sumofDifferences > 0) {
	    	sumofDifferencesAllRuns += sumofDifferences;
	    	sumofAvgDifferencesAllRuns += sumofAvgDifferences; sumofRandomDifferencesAllRuns += sumofRandomDifferences;
	    	sumofBenchmarkADifferencesAllRuns += sumofBenchmarkADifferences;
	    	sumofBenchmarkBDifferencesAllRuns += sumofBenchmarkBDifferences;
		}
		//"Subset\tFinalAvgLoss\tsumofDifferences\tsumofRandomDifferences\tsumofBenchmarkADifferences\tsumofBenchmarkBDifferences"
		output11 << a << '\t' << finalAvgLoss << '\t' << sumofDifferences << '\t' << sumofRandomDifferences << '\t' << sumofBenchmarkBDifferences << '\t' << sumofBenchmarkBDifferences << endl;

	    inputfile.close();

	    cout << "Closed input file" << endl;

	    if(false) {
		    string nnwfilename = "nnwfile.nnw"; nnwfilename += to_string(a);
		    cout << "Writing neural network to file: " << nnwfilename << endl;
		    ofstream nnwfile(nnwfilename);
		    nnw -> writeWeights(nnwfile);
		    nnwfile.close();
		}


		cout << "Attempting to delete nnw" << endl;
		//cout << "nnw is pointing to " << nnw << endl;
	    delete(nnw);

	    cout << "nnw deleted succesfully" << endl;
	}
	cout << "Over " << runSize << " trials:" << endl;
	cout << setw(30) << "SUM OF DIFFERENCES : " << sumofDifferencesAllRuns << endl;
	cout << setw(30) << "SUM OF AVG. DIFFERENCES" << sumofAvgDifferencesAllRuns << endl;
	cout << setw(30) << "SUM OF RANDOM DIFFERENCES" << sumofRandomDifferencesAllRuns << endl;
	cout << setw(30) << "SUM OF BENCHMARK A DIFFERENCES" << sumofBenchmarkADifferencesAllRuns << endl;
	cout << setw(30) << "SUM OF BENCHMARK B DIFFERENCES" << sumofBenchmarkBDifferencesAllRuns << endl;

	/*cout << "Writing diffs to file: " << outputfilename << endl;
    output.close();
    cout << "Writing avg diffs to file: " << output2filename << endl;
    output2.close();
    cout << "Writing random diffs to file: " << output3filename << endl;
    output3.close();
    cout << "Writing outputs,expecteds diffs to file: " << output4filename << endl;
    output4.close();

    //cout << "Writing avg. diffs of tested data per subset to file: " << output5filename << endl;
    output5.close();
    //cout << "Writing random. diffs tested on nnw per subset to file: " << output6filename << endl;
    output6.close();
    cout << "Writing the dataset to: " << output7filename << endl;
    output7.close();
    cout << "Writing final avg. loss for each subset to file: " << output8filename << endl;
    output8.close();*/

    //cout << "Writing info to file: " << output9filename << endl;
    double predictionScoreRandom = sumofRandomDifferencesAllRuns / sumofDifferencesAllRuns;
    double predictionScoreBenchmarkA = sumofBenchmarkADifferencesAllRuns / sumofDifferencesAllRuns;
    double predictionScoreBenchmarkB = sumofBenchmarkBDifferencesAllRuns / sumofDifferencesAllRuns;
    //output9 << "Prediction Score: "; output9 << predictionScore; output9 << endl; output9 << endl;
    //output9 << "SUM OF Diferences: "; output9 << sumofDifferencesAllRuns; output9 << endl;
    //output9 << "SUM OF RANDOM DIFFERENCES: "; output9 << sumofRandomDifferencesAllRuns; output9 << endl;


    //output9.close();

    

    //RUN the RSCRIPT
    if(true) {
    	string command = "Rscript benchmark.R " + extension + " " + argv[1];
    	int result = system(command.c_str());
    	cout << " SYSTEM: Ran rscript with system command: " << command << endl;
    	cout << " SYSTEM Command returns:" << result << endl;
    	//Rscript benchmark.R plantdataL_1 plantdatajobL.txt
    }

    cout << "Updating the prediction score data file " << psDataname << endl;

    stringstream run_summary;

    //"Dataset	Filesuffix	RunSize	TestsperRun	Stochastic?	...Layersizes...	numTrainings	RandomPredictionScore	BenchmarkAPredictionScore	BenchmarkBPredictionScore	Notes"
    run_summary << inputfilename << "\t" << extension << "\t" << runSize << "\t" << testSize << "\t" << stochastic << "\t";
    for(int i = 0; i < numInternalCols; i++) {
    	if(i == 0 && filtersize != 0) {
    		run_summary << "FILTER(" << filtersize << ");LAYERS(" << internalCols[i] << "):";
    	} else {
    		run_summary << internalCols[i]; if(i < numInternalCols - 1) run_summary << ":";
    	}
    }run_summary << "\t";
    run_summary << totalWeights << "\t";
    run_summary << runTimes << "\t" << predictionScoreRandom << "\t" << predictionScoreBenchmarkA << "\t" << predictionScoreBenchmarkB << "\t";

    if(predictionScoreRandom > bestPredictionScore) {
    	run_summary << "NewBest";
    } else run_summary << "n/a";


    //run_summary << endl;

    psData << run_summary.str() << endl;
    cout << "Wrote: " << run_summary.str() << " to the psData file:" << psDataname << endl;
    output12 << run_summary.str() << endl;

    cout << "Wrote info about every test to " << output10filename << endl;
    cout << "Wrote info about every subset to " << output11filename << endl;
    cout << "Wrote info about parameters and results to " << output12filename << endl;

    output10.close();
    output11.close();
    output12.close();

    //
    if(mutate) {
	    ofstream jobfilewrite(argv[1], ios::out);
		if(mutate) {
			cout << "Writing to the jobfile: " << endl;
		// "dataset,filesuffix,run size,partition size,stochastic?,numIcols,filtersize,layer sizes,total weights,trainings,Prediction Score"	
			//3. write the best hyperparamers to the job file, update the best prediction score if neccesary
			//ofstream joboutputfile(argv[1]);
			//extension.erase(extension.size()-1);

			if(predictionScoreRandom > bestPredictionScore) {
				cout << "Writing new pararmters to the jobfile..." << endl;
				//jobfilewrite << inputfilename << " "; cout << inputfilename << " "; cout << inputfilename;
				jobfilewrite << originalExtension << jobID << " "; cout << originalExtension << jobID << " ";

				//jobfilewrite << runSize << " ";
				//jobfilewrite << testSize << " ";
				jobfilewrite << stochastic << " ";
				jobfilewrite << numInternalCols << " ";
				//jobfilewrite << filtersize << " ";

				cout << runSize << " ";
				cout << testSize << " ";
				cout << stochastic << " ";
				cout << numInternalCols << " ";
				//cout << filtersize << " ";
				for(int i = 0; i < numInternalCols; i++) {
					jobfilewrite << internalCols[i] << " ";
					cout << internalCols[i] << " ";
				}
				jobfilewrite << runTimes << endl;
				cout << runTimes << endl;

				if(jobfilewrite) {
					cout << "File stream still good" << endl;
				} else 
				cout << "File stream NOT good" << endl;

				// update prediction score
				bestPredictionScore = predictionScoreRandom;
			} else {
				cout << "Writing the old parameters to the jobfile..." << endl;
				//jobfilewrite << inputfilename << " ";
				jobfilewrite << originalExtension << jobID << " ";

				//jobfilewrite << runSize << " ";
				//jobfilewrite << BRtestSize << " ";
				jobfilewrite << BRstochastic << " ";
				jobfilewrite << BRnumInternalCols << " ";
				//jobfilewrite << BRfiltersize << " ";
				for(int i = 0; i < BRnumInternalCols; i++) {
					jobfilewrite << BRinternalCols[i] << " ";
				}
				jobfilewrite << BRrunTimes << endl;
			}
			jobID ++;
		}
		jobfilewrite.close();
		//
		jobfile.close();
	}

}
psData.close();

	return 0;
}