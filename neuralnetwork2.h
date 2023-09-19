// Neural Network

// 
// 
#include <vector>
#include <string>
#include <iostream>
#include <math.h>
#include <cmath>
#include <time.h>  
#include <fstream>
#include <iomanip>
using namespace std;

// forward declerations
class Node;
class Column;
class Input;

class Functions{
public:
	static double sigmoidCoefficient;
	static double sigmoidStretch;
	static double linearCoefficient;
	static double constantBias;
	static double rectifiedLinearActivationSlope;
public:
	static char intToACGT(int n) {
		char val = 'Z';
		if(n == 0) val = 'A'; else if (n == 1) val = 'C'; else if (n == 2) val = 'G'; else if (n == 3) val = 'T'; else if (n == 4) val = '-';
		return val;
	}
	static int ACGTtoInt(char c) {
		int val = -1;
		if(c == 'A') val = 0; else if (c == 'C') val = 1; else if (c == 'G') val = 2; else if (c == 'T') val = 3; else if (c == '-') val = 4;
		return val;
	}
	static bool Debug() {return false;}
	static bool Stochastic() {return false;}
	static double SigMoid(double x) {
		//return Functions::rectifiedLinearActivation(x);
		double coefficient = Functions::sigmoidCoefficient;
	  //cout << "calculating sigmoid of : " << x << " as " << "" << endl;
		return Functions::sigmoidStretch*(1.0 / (1.0+exp(-1*coefficient*x))) + linearCoefficient*x + Functions::constantBias;

	}
	static double SigMoidDerivative(double x) {
		//return Functions::rectifiedLinearActivationDerivative(x);
		double coefficient = Functions::sigmoidCoefficient;
		// cout << "calculating sigmoid of derivative: " << x << " as " << "" << endl;
		return Functions::sigmoidStretch*((coefficient * exp(-1*coefficient*x)) / pow(1 + exp(-1*coefficient*x), 2)) + linearCoefficient;
		// 3e^(-3x) / (1 + e^(-3x))^2
	}
	static double rectifiedLinearActivation(double x) {
		if(x > 0) {
			//cout << "returning " << x*Functions::rectifiedLinearActivationSlope << endl;
			return x*Functions::rectifiedLinearActivationSlope;
		}
		// leaky version
		else return x*Functions::rectifiedLinearActivationSlope * 0.1;
	}
	static double rectifiedLinearActivationDerivative(double x) {
		if (x > 0) return Functions::rectifiedLinearActivationSlope;
		else return Functions::rectifiedLinearActivationSlope * 0.1;


	}
};
double Functions::sigmoidCoefficient = 1;
double Functions::sigmoidStretch = 1;
double Functions::linearCoefficient = 0.001;
double Functions::constantBias = 0;
double Functions::rectifiedLinearActivationSlope = 0.1;
class Node{
  public:
	// weight: How much it will activate all of the next neurons in the next col (AKA tensors)
	double* weights;
	int connections; // number of neurons in the next column
	double activation;
	double nonSigmoidactivation;

	double* weightdldwSum;
	int count = 0; // counts how many times weighteddldwSum is added to
	Column* next_column;
	// this array holds the sum of each dl/dw for each weight this node pointing to the next layer
	
	// this will be based on how much it is activated * its weight to that node
	// the weights can be negative or positive

	double Y;
	double constantBias;

	double nodedldw;
	bool hasCalculatedNode;

	~Node() {
		delete(weights);
		delete(weightdldwSum);
	}

	Node(int n, Column* next) {
		connections = n;
		weights = new double[connections];
		weightdldwSum = new double[connections];
		for(int i = 0; i < n; i ++) {
			// the strength of each weight is random at the beginning
			weights[i] = ((rand() % 100) / 50.0) - 1; // random between -1 and 1
			//cout << (rand() % 100) / 2.0 << endl;
		} for(int i = 0; i < n; i ++ ) {
			weightdldwSum[i] = 0;
		}
		next_column = next;
		Reset();
	}
	void Reset() {activation = 0; nonSigmoidactivation = 0; Y = 0; nodedldw = 0; hasCalculatedNode = false;}
	void ApplySigmoid() {
		nonSigmoidactivation = activation;
		activation = Functions::SigMoid(activation);
		
	}
	
	void DisplayBiases() {
		cout << " Activation:" << activation << " nonSigmoidactivation:" << nonSigmoidactivation << " Weights: ";
		for(int i = 0; i < connections; i++) {
			//cout << i << ": " << weights[i] << " dl/dwsum:" << weightdldwSum[i] << " ";
			cout << i << ": " << weights[i] << " dl/dwsum:" << weightdldwSum[i];
		} cout << endl;
	}
	void CalculateWeightdldws();
	
	void getExpectedOutput(double expected) {
		// nodes in the output layer need to know Y, as they need it for recursively helping find the
		// dl/dw for all of the previous weights as it is used in the formula
		Y = expected;
	}
	double getNodedldw();
	void AdjustWeights(double factor) {
		for(int i = 0; i < connections; i++) {
			// consider the weightdldwSum = 0.05. So for every 1 we increase the weight the loss increases be 0.05.
			// SO we want to decrease the weight
			weights[i] += weightdldwSum[i] * factor / (double)count; // dividing by count accounts for how many data were summed up
			//cout << "   >>> Increasing connection by " << weightdldwSum[i] << endl;
			weightdldwSum[i] = 0;
		}
		count = 0;
	}
	void writeWeights(ofstream& nnwfile) {
		for(int i = 0; i < connections; i++) {
			nnwfile << weights[i] << endl;
		}
	}
	void readWeights(ifstream& nnwfile) {
		for(int i = 0; i < connections; i++) {
			nnwfile >> weights[i];
		}
	}
	
};

class Column{
  public:
	// Represents one vertical level (layer) of neurons in the network, other than the input layer
	Column* next_column;
	Node** nodes; 
	int numNodes;

	~Column() {for(int i = 0; i < numNodes; i++) delete(nodes[i]); delete(nodes);}

	Column(int n, Column* next) {
		// Initialize a normal Layer
		numNodes = n;
		nodes = new Node*[numNodes];
		next_column = next;
		if(next) {
			// Initialize all nodes and their weights
			for(int i = 0; i < numNodes; i++) {
				nodes[i] = new Node(next_column -> numNodes, next_column);
			}

		} else {
			// this is the output layer!
			// initialize all of the nodes with no connections
			for(int i = 0; i < numNodes; i++) {
				nodes[i] = new Node(0, next_column);
			}
		}
	}
	
	int howManyWeights() {
		int weightCount = numNodes * next_column -> numNodes;
		return weightCount;
	}
	void Reset() {
		for(int i = 0; i < numNodes; i++) {
			nodes[i] -> Reset();
		}
		//next_column -> Reset();
	} 
	void ApplySigmoid() {
		for(int i = 0; i < numNodes; i++) {
			nodes[i] -> ApplySigmoid();
		}
	}
	void Display() {
		for(int i = 0; i < numNodes; i ++) {
			cout << "Node " << i << ": ";
			nodes[i] -> DisplayBiases();
		}
	}
	void DisplayActivations() {
		for(int i = 0; i < numNodes; i ++) {
			cout << "Node " << i << " Activation : ";
			cout << nodes[i] -> activation << endl;
			cout << "Node " << i << " Activation before Sigmoid : ";
			cout << nodes[i] -> nonSigmoidactivation << endl;
		}
	}
	void CalculateNodeDlDws() {
		for(int i = 0; i < numNodes; i++) {
			if (Functions::Debug()) cout << "Finding dl/dw for all of the weights in node in row " << i << "..." << endl;
			nodes[i] -> CalculateWeightdldws();
		}
	}
	void FeedYValues(double expected) {
		for(int i = 0; i < numNodes; i++) {
			nodes[i] -> getExpectedOutput(expected);
		}
	}
	void AdjustWeights(double factor) {
		for(int i = 0; i < numNodes; i++) {
			nodes[i] -> AdjustWeights(factor);
		}
	}
	void writeWeights(ofstream& nnwfile) {
		for(int i = 0; i < numNodes; i++) {
			nodes[i] -> writeWeights(nnwfile);
		}
	}
	void readWeights(ifstream& nnwfile) {
		for(int i = 0; i < numNodes; i++) {
			nodes[i] -> readWeights(nnwfile);
		}
	}
};

class Input{
  public:
	// The original column which will have 4x the number of nodes as inputs
	Column* next_column;
	Node*** nodes;
	int numInputs;

	// extra info for convolutional layers
	bool convolutional = false;
	//Node*** cNodes;
	double*** convWeights;
	double*** convWeightsdldwsum;

	int filtersize;
	int count = 0; // count how many times backweights has been adjusted
	int numSectors;
	int nodesPerSector; // how many nodes are in the next layer. The actual number is nodesPerSector * numSectors.

	~Input() {
		for(int i = 0; i < 4; i++) {
			for(int j = 0; j < numInputs; j++) {
				delete nodes[i][j];
			}
			delete[] nodes[i];
		}
		delete[] nodes;
		//Display();
		if(convolutional) {
			for(int i = 0; i < numSectors; i++) {
				for(int j = 0; j < 4; j++) {
					delete[] convWeights[i][j];
					delete[] convWeightsdldwsum[i][j];
				}
				delete[] convWeights[i];
				delete[] convWeightsdldwsum[i];
			}
			delete[] convWeights;
			delete[] convWeightsdldwsum;
		}
	}
	Input(int n, Column* next) { // constructor 
		convolutional = false;
		cout << "NNW" << endl;
		// create structure
		numInputs = n;
		nodes = new Node**[4];
		next_column = next;
		for(int i = 0; i < 4; i ++){
			nodes[i] = new Node*[numInputs];
			// fill in the structure with nodes
			for (int j = 0; j < numInputs; j++) {
				nodes[i][j] = new Node(next_column -> numNodes, next_column);
			}
		}
		// nodes -> [A C G T]
		// each of these point onto an array of pointers to Nodes  
		Reset();
	}
	Input(int n, Column* next, int fs) {
		// Initialize a convolutional Input Layer
		cout << "CNN" << endl;
		convolutional = true;
		filtersize = fs;
		numInputs = n;
		nodesPerSector = numInputs + 1 - filtersize; // NEW

		nodes = new Node**[4];
		next_column = next;
		for(int i = 0; i < 4; i ++){
			nodes[i] = new Node*[numInputs];
			// fill in the structure with nodes
			for (int j = 0; j < numInputs; j++) {
				nodes[i][j] = new Node(next_column -> numNodes, next_column);
			}
		}

		if((next_column -> numNodes % nodesPerSector) == 0) {
			numSectors = (next_column -> numNodes / nodesPerSector);
			cout << "numsectors, nextcol nodes = " << next_column -> numNodes << " ; nodesPerSectors = " << nodesPerSector << " implies numSectors = " << numSectors << endl;
			// 
		} else {
			cout << "The first input column is poorly configured, it has " << next_column -> numNodes << " nodes, there is expected to be " << nodesPerSector << " nodes per sector" << endl;
			exit(0);
		}

		cout << "numSectors initializing: " << numSectors << endl;
		
		convWeights = new double**[n];
		convWeightsdldwsum = new double**[n];
		for(int i = 0; i < numSectors; i++) {
			convWeights[i] = new double*[4];
			convWeightsdldwsum[i] = new double*[4];
			for(int j = 0; j < 4; j++) {
				convWeights[i][j] = new double[filtersize];
				convWeightsdldwsum[i][j] = new double[filtersize];
				for(int k = 0; k < filtersize; k++) {
					convWeights[i][j][k] = ((rand() % 100) / 50.0) - 1; // random between -1 and 1
					convWeightsdldwsum[i][j][k] = 0;
				}
			}
		}
		Reset();

	}
	void Reset() {
		if(convolutional) {
			for(int h = 0; h < numSectors; h++) {
				for(int k = 0; k < filtersize; k++) {
					for(int j = 0; j < 4; j++) {
						convWeightsdldwsum [h][j][k] = 0;
					}
				}
			}
			count = 0;
		}

		for(int i = 0; i < 4; i++) {
			for(int j = 0; j < numInputs; j++) {
				nodes[i][j] -> Reset();
			}
		}
		
		//next_column -> Reset();
	}
	void Display() {
		if(!convolutional) {
			cout << "Regular NNW" << endl;
			for(int i = 0; i < numInputs; i ++){
				cout << "Index " << i << " : " << endl;
				// fill in the structure with nodes
				for (int j = 0; j < 4; j++) {
					cout << Functions::intToACGT(j) << ": ";
					nodes[j][i] -> DisplayBiases();
					//cout << endl;
				}
			}
		}
		else {
			cout << "CNN" << endl;
			for(int h = 0; h < numSectors; h++) {
				cout << "Sector " << h << " : " << endl;
				for(int j = 0; j < 4; j++) {
					cout << "  Nucleotide " << Functions::intToACGT(j) << ":";
					for(int k = 0; k < filtersize; k++) {
						cout << "  Filter Pos " << k << " weight:" << convWeights[h][j][k] << ",dldwsum:" << convWeightsdldwsum[h][j][k];
					}
					cout << endl;
				}
			}

		}
	}
	void CalculateNodeDlDws() {
		if(convolutional) {
			// calculate the convWeight parameter gradients instead BACKHERE
			for(int h = 0; h < numSectors; h++) {
				for(int k = 0; k < filtersize; k++) {
					for(int j = 0; j < 4; j++) {
						//cout << "Calculating the DlDw in sector " << h << " for the filter index " 
						//<< k << " base " << Functions::intToACGT(j) << "..."<< endl;
						// calculate the DlDw of this weight;
						double sum = 0;
						for(int m = 0; m < nodesPerSector; m++) {
							//if(k + m < numNodes) 
							if(nodes[j][k + m] -> activation == 1) {
								sum += next_column -> nodes[(h * nodesPerSector) + m] -> getNodedldw();
								//cout << "  Adding from node " << k+ m << " in the backlayer: " << "dldwSum increased by: " << nodes[h][m] ->getNodedldw() << endl;
							}
						}
						double dldw = sum;
						//cout << "This weight is from an activation of " << activation << " which is multiplied by " << next_column -> nodes[i] -> getNodedldw();
						//cout << " to get a calculation of " << dldw <<" ... ";
						//cout << "   >>In total, this training data changed this weight's dldw by: " << dldw << endl;
						convWeightsdldwsum[h][j][k] += dldw;
						//cout << "dl/dw sum so far: " << backWeightdldwSum[h][j][k] << endl;
					}
					
				}
			}
			count++;

		} else {
			for(int i = 0; i < 4; i++) {
				for(int j = 0; j < numInputs; j++) {
					if (Functions::Debug()) cout << "Finding dw/dl for all of the weights in node in col " << Functions::intToACGT(i) << " row " << j << "..." << endl;
					nodes[i][j] -> CalculateWeightdldws();
				}
			}
		}

	}
	void AdjustWeights(double factor) {
		if(convolutional) {
			//cout << "ADJUSTING CONVOLUTONAL WEIGHTS..." << endl;
			for(int h = 0; h < numSectors; h++) {
				for(int k = 0; k < filtersize; k++) {
					for(int j = 0; j < 4; j++) {
						if(isnan(convWeightsdldwsum[h][j][k]))cout << "When adjusting convWeights, sum in NAN" << endl;
						convWeights[h][j][k] += convWeightsdldwsum [h][j][k] * factor / (double)count;
						if(count == 0) cout << "BACKWEIGHT divide by 0" << endl;
						//cout << "   >>> Increasing connection by " << backWeightdldwSum [h][j][k] * factor / (double)nodes[h][0] -> count << endl;
					}
				}
			}
		}
		else {
			for(int i = 0; i < 4; i++) {
				for(int j = 0; j < numInputs; j++) {
					nodes[i][j] -> AdjustWeights(factor);
				}
			}
		}
	}
	void writeWeights(ofstream& nnwfile) {
		if(convolutional) {
			cout << "To IMPLEMENT: write CNN weights to file" << endl;
			exit(1);
		}
		for(int i = 0; i < 4; i++) {
			for(int j = 0; j < numInputs; j++) {
				nodes[i][j] -> writeWeights(nnwfile);
			}
		}
	}
	void readWeights(ifstream& nnwfile) {
		if(convolutional) {
			cout << "To IMPLEMENT: read CNN weights from file" << endl;
			exit(1);
		}
		for(int i = 0; i < 4; i++) {
			for(int j = 0; j < numInputs; j++) {
				nodes[i][j] -> readWeights(nnwfile);
			}
		}
	}
	int howManyWeights() {
		int weightCount = numInputs * 4 * next_column -> numNodes;
		if(convolutional)
			weightCount = numSectors * filtersize * 4;
		return weightCount;
	}
};

inline void Node::CalculateWeightdldws() {
	for(int i = 0; i < connections; i++) {
		// calculate the DlDw of this weight;
		// returns the activation of this node + the nodedldw of the node the weight it points to
		double dldw = activation * next_column -> nodes[i] -> getNodedldw();
		//cout << "This weight is from an activation of " << activation << " which is multiplied by " << next_column -> nodes[i] -> getNodedldw();
		//cout << " to get a calculation of " << dldw <<" ... ";
		weightdldwSum[i] += dldw;
		if(Functions::Debug())cout << "dl/dw for weight " << i << " is calculated as: " << weightdldwSum[i] << endl;
		
	}
	count++;
}
inline double Node::getNodedldw() {
  if(!hasCalculatedNode) {
		// algorithm for when it is an internal node (node in the output column, i.e. next_column has a value)
		if(Functions::Debug())cout << "calling getNodedldw on a node... ";
		if(next_column) {
			if(Functions::Debug())cout << "calling getNodedldw on an internal node " << endl;
			double sum = 0;
			for(int i = 0; i < connections; i++) {
				// add the getNodedldw for the all of the nodes this node points to * the weights
				sum += next_column -> nodes[i] -> getNodedldw() * weights[i];
			}
			// return this functions activation WITHOUT sigmoid applied to the sigmoid derivative function * the sum from before
			nodedldw = Functions::SigMoidDerivative(nonSigmoidactivation) * sum;
		} else {
			//if(Functions::Debug())cout << " > For the final output node, returning " << 2 * (Y - activation) * Functions::SigMoidDerivative(nonSigmoidactivation) << endl;
			nodedldw = 2 * (Y - activation) * Functions::SigMoidDerivative(nonSigmoidactivation);

			// this is for when this is the part of the output node column. The object must know what the expected output (Y) is.
			// returns 2(Y - activation) * SigmoidDerivative(nonSigmoidactivation);
		}
		hasCalculatedNode = true;
	}
	
	return nodedldw;
}


class NeuralNetwork{
  public:

	Column* outputs;
	//Column* col2;
	//Column* col1;
	Input* ip;

	// stores the sizes, including of the input and output layer
	vector<int> colSizes;
	// stores the pointers for the internal layers
	Column** internalCols;

	int val;
	int numICols;

	int filtersize = 0;
	int numSectors = 0;
	bool convolutional = false;
  public:
  	~NeuralNetwork() {
  		delete(ip);
  		delete(outputs);
  		for(int i = 0; i < numICols; i++) delete(internalCols[i]);
  		delete[] internalCols;
  	}
	void createLayers() {
		outputs = new Column(colSizes[numICols + 1], 0);
	  internalCols = new Column*[numICols];
		for(int i = numICols - 1; i > -1; i--) {
			if(i == numICols - 1) {
				internalCols[i] = new Column(colSizes[i + 1], outputs);
			} 
			else {
				internalCols[i]= new Column(colSizes[i + 1], internalCols[i + 1]);
			}
			// each column points to the next one
		}
		//for(int i = 0; i < (int)ic.size()-1; i++) {cout << internalCols[i] << endl;}
		// the input points to the first one

		if(numICols > 0) {
			if(filtersize > 0)
				ip = new Input(colSizes[0], internalCols[0], filtersize);
			else
				ip = new Input(colSizes[0], internalCols[0]);
		}
		
		// unless there are no internal layers; it points to the output directly
		else {
			if(filtersize > 0)
				ip = new Input(colSizes[0], outputs, filtersize);
			else
				ip = new Input(colSizes[0], outputs);
		}
		
	}
	NeuralNetwork(ifstream& nnwfile) {
		readWeights(nnwfile);
	}
	NeuralNetwork(int a, vector<int>&ic, int d) {
		cout << "NNW initializing" << endl;
		convolutional = false;
		//exit(0);
		numICols = (int)ic.size();
		colSizes.push_back(a);
		for(auto i : ic) {colSizes.push_back(i);}
		colSizes.push_back(d);
		createLayers();
	}
	NeuralNetwork(int a, int fs, vector<int>ic, int d) {
		cout << "CNN initializing" << endl;
		//exit(0);
		convolutional = true;
		filtersize = fs;
		//cout << "ic[0] (number of sectors) = " << ic[0] << ", a + 1 - filtersize = " << a + 1 - filtersize << endl;
		ic[0] = (a + 1 - filtersize) * ic[0];
		//cout << "Convolutional layer will have " << ic[0] << " nodes" << endl;
		numICols = (int)ic.size();
		colSizes.push_back(a);
		for(auto i : ic) {colSizes.push_back(i);}
		colSizes.push_back(d);
		createLayers();
	}
	void display() {
		cout << "Input layer: " << endl;
		ip->Display();
		Column* curCol = ip->next_column;
		int i = 1;
		while(curCol -> next_column) {
			cout << "Layer " << i << ": " << endl;
			curCol -> Display();
			curCol = curCol -> next_column;

			i++;
		}
	}
	double runInput(string input, double expected, bool silent) {
		Column* curCol = ip -> next_column;
		Column* nextCol;
		//cout << "Running with input " << input << endl;
		// eg. "AAC"
		// 0) reset the activation values of the nodes
		ip -> Reset(); // Note: recurses throughout the network
		while(curCol) {
			curCol->Reset(); curCol = curCol -> next_column;
		}

		// 1) ip: put 1 to the activation values of the nodes who have the value of the input string
		nextCol = ip->next_column;
		for(int i = 0; i < ip->numInputs; i++) {
			char c  = input[i];
			val = Functions::ACGTtoInt(c);
			if(val != 4)
			ip -> nodes[val][i] -> activation = 1;



			// assign the next col node values here (all other nodes will have activation of 0)
			if(val != 4)
			if(!convolutional)
			for(int j = 0; j < nextCol -> numNodes; j++) {
				// increase the activation of all of the nodes it points to based solely on the weights (activations are all 1)
				nextCol -> nodes[j] -> activation += ip -> nodes[val][i] -> weights[j]; 
				//cout << "Increasing the activation of node " << j << " in col 1 by input at row, col of ip neuron array of indices " << val << ", " << i << ": "
				//<< ip -> nodes[val][i] -> weights[j] << endl;
			}
			
			if(Functions::Debug()) nextCol -> DisplayActivations();
		}
		if(convolutional) {
			// assign the first layer's activations based on the convolutional weights of the input layer
			for(int h = 0; h < ip -> numSectors; h++) {
				for(int m = 0; m < ip -> nodesPerSector; m++) {
					for(int k = 0; k < ip -> filtersize; k++) {
						for(int j = 0; j < 4; j++) {
							if(ip -> nodes[j][m + k] -> activation == 1)
						// increase the activatin of the node in first column Weights [sector][nucleotide][filtersize]
								nextCol -> nodes[(h * ip -> nodesPerSector) + m] -> activation += ip -> convWeights[h][j][k];
						}

					}
				}
			}
		}

		nextCol -> ApplySigmoid();

		//
		curCol = ip -> next_column;
		while(curCol -> next_column) {
			nextCol = curCol -> next_column;
			// increase the activation off all nodes this col points to
			for(int i = 0; i < curCol->numNodes; i++) {
			// increase all of the col2 for that node
				for(int j = 0; j <  nextCol -> numNodes; j++) {
					// increase activation of col2 nodes by: activation of col1 * weight
					nextCol -> nodes[j] -> activation += curCol -> nodes[i] -> activation * curCol -> nodes[i] -> weights[j];
					//cout << "Increasing the activation of node " << j << " in nextCol by input at row of curCol neuron array at index " << i << ": "
					//<< curCol -> nodes[i] -> activation * curCol -> nodes[i] -> weights[j] << endl;
				}
				if(Functions::Debug()) nextCol -> DisplayActivations();
			}

			curCol -> next_column -> ApplySigmoid();
			curCol = curCol -> next_column;
		}
		// print out the activations of all of the output nodes' activations
		for(int i = 0; i < outputs -> numNodes; i ++) {
			if(!silent)cout << setw(11) << outputs -> nodes[i] -> activation << " " ;	
			//<< setw(11) << outputs -> nodes[i] -> nonSigmoidactivation << " "
		}
		double loss = pow(outputs -> nodes[0] -> activation - expected, 2);
		if(expected==-1) loss = 0;
		
		if(!silent && expected!=-1) {
			cout << "Expected: " << left << setw(10) << expected << " ";
			if(loss > 0.04) cout << left << setw(6) << "\033[1;31m" << "Loss = " << setw(11) << loss << "(!!!)" << "\033[0m"; //diff > 0.2
			else if(loss > 0.0025) cout << left << setw(5) << "\033[31m" << "Loss = " << setw(11) << loss << "(!)" << "\033[0m";
			else cout << left << setw(6) << "Loss = " << setw(11) << loss;
			cout << endl;
		}

		outputs -> FeedYValues(expected);

		//cout << "Calculating dL / dw for all input weights" << endl;
		// 1) call calculate dL / dw on input object.
		// 2) this object calls calculateAllWeightdl/dw on all of its nodes (in this case the 80 nodes before the internal layer).
		// 3) each of these nodes will calculate all of the dL/dw for all of its weights in order and store those in an array.
		// 4) calculate weight dL/dw returns the activation of the node * getNodedl/dw on the (singular) node it points to.
		// 5) the function getNodedl/dw called on a specific Node is the recursive method. 
		//     >> this function returns the sum, for EACH node it points to in the next layer, of the WEIGHT to that node * getNodedl/dw on it
		//        and also * the derivative of the sigmoid function applied to the activation of that node WITHOUT the sigmoid function applied to it 
		//        (i.e. I, where I = a1w1 + a2w2 +... NOT: Activation = sigma(I) = sigma(a1w1 + a2w2 +...) (note this multiplication is done AFTER the sum as it is the same for each term) 
		//
		//     >> this function recursively terminates at the final output node, where it returns instead the derivative of the sigmoid function(I)
		//              * 2 (Y - A) [as this is the derivative of the loss function]

		if(Functions::Debug())cout << "CALCULATE IP GRADIENT" << endl;
		ip -> CalculateNodeDlDws();

		curCol = ip->next_column;
		while(curCol->next_column) {
			if(Functions::Debug())cout << "CALCULATE INTERNAL GRADIENT" << endl;
			// recurses through the network for each layer and gets the dl/dws
			curCol->CalculateNodeDlDws();
			curCol = curCol -> next_column;
		}

		return loss;
	}
	void AdjustWeights(bool silent) {
		if(!silent)cout << " >> Adjusting Weights for cols and inputs..." << endl;

		ip -> AdjustWeights(1);
		Column* curCol = ip->next_column;
		while(curCol->next_column) {
			curCol->AdjustWeights(1);
			curCol = curCol -> next_column;
		}

	}
	void writeWeights(ofstream& nnwfile){
		for(auto a : colSizes) {
			nnwfile << a << " ";
			
		}
		nnwfile << "-1" << endl;

		ip -> writeWeights(nnwfile);
		Column* curCol = ip->next_column;
		while(curCol->next_column) {
			curCol->writeWeights(nnwfile);
			curCol = curCol -> next_column;
		}
	}
	void readWeights(ifstream& nnwfile) {
		// get the structure
		int a = 0;
		while(a != -1) {
			nnwfile >> a;
			if(a != -1)
			colSizes.push_back(a);
		}
		//nnwfile >> a;
		
		numICols = (int)colSizes.size() - 2;
		createLayers();
		// set the correct amounts

		ip -> readWeights(nnwfile);
		Column* curCol = ip->next_column;
		while(curCol->next_column) {
			curCol->readWeights(nnwfile);
			curCol = curCol -> next_column;
		}
	}
	int howManyWeights(){
		int totalWeights = 0;

		totalWeights += ip -> howManyWeights();

		Column* curCol = ip->next_column;
		while(curCol->next_column) {
			totalWeights += curCol -> howManyWeights();
			curCol = curCol -> next_column;
		}
		return totalWeights;

	}

};