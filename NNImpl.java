import java.util.*;

/**
 * The main class that handles the entire network
 * Has multiple attributes each with its own use
 */

class NNImpl {
    private ArrayList<Node> inputNodes;     //list of the input layer nodes.
    private ArrayList<Node> hiddenNodes;    //list of the hidden layer nodes
    private ArrayList<Node> outputNodes;    // list of the output layer nodes

    private ArrayList<Instance> trainingSet;    //the training set

    private double learningRate;    // variable to store the learning rate
    private int maxEpoch;   // variable to store the maximum number of epochs
    private Random random;  // random number generator to shuffle the training set

    /**
     * This constructor creates the nodes necessary for the neural network
     * Also connects the nodes of different layers
     * After calling the constructor the last node of both inputNodes and
     * hiddenNodes will be bias nodes.
     */

    NNImpl(ArrayList<Instance> trainingSet, int hiddenNodeCount, Double learningRate, int maxEpoch,
           Random random, Double[][] hiddenWeights, Double[][] outputWeights) {
        this.trainingSet = trainingSet;
        this.learningRate = learningRate;
        this.maxEpoch = maxEpoch;
        this.random = random;

        //input layer nodes
        this.inputNodes = new ArrayList<>();
        int inputNodeCount = trainingSet.get(0).attributes.size();
        int outputNodeCount = trainingSet.get(0).classValues.size();
        for (int i = 0; i < inputNodeCount; i++) {
            Node node = new Node(Constants.INPUT);
            this.inputNodes.add(node);
        }

        //bias node from input layer to hidden
        Node biasToHidden = new Node(Constants.HIDDEN_BIAS);
        this.inputNodes.add(biasToHidden);

        //hidden layer nodes
        this.hiddenNodes = new ArrayList<>();
        for (int i = 0; i < hiddenNodeCount; i++) {
            Node node = new Node(Constants.HIDDEN);
            //Connecting hidden layer nodes with input layer nodes
            for (int j = 0; j < this.inputNodes.size(); j++) {
                NodeWeightPair nwp = new NodeWeightPair(this.inputNodes.get(j), hiddenWeights[i][j]);
                node.parents.add(nwp);
            }
            this.hiddenNodes.add(node);
        }

        //bias node from hidden layer to output
        Node biasToOutput = new Node(Constants.OUTPUT_BIAS);
        this.hiddenNodes.add(biasToOutput);

        //Output node layer
        this.outputNodes = new ArrayList<>();
        for (int i = 0; i < outputNodeCount; i++) {
            Node node = new Node(Constants.OUTPUT);
            //Connecting output layer nodes with hidden layer nodes
            for (int j = 0; j < this.hiddenNodes.size(); j++) {
                NodeWeightPair nwp = new NodeWeightPair(this.hiddenNodes.get(j), outputWeights[i][j]);
                node.parents.add(nwp);
            }
            this.outputNodes.add(node);
        }
    }

    /**
     * Get the prediction from the neural network for a single instance
     * Return the idx with highest output values. For example if the outputs
     * of the outputNodes are [0.1, 0.5, 0.2], it should return 1.
     * The parameter is a single instance
     */

    public int predict(Instance instance) {
        runNeuralNet(instance);
        int prediction = 0;
        double max = Double.MIN_VALUE;
        for(int i = 0; i < this.outputNodes.size(); i++) {
            double out = this.outputNodes.get(i).getOutput();
            if(out > max) {
                max = out;
                prediction = i;
            }
        }
        return prediction;
    }


    /**
     * Train the neural networks with the given parameters
     * <p>
     * The parameters are stored as attributes of this class
     */

    public void train() {
        for(int j = 0; j < this.maxEpoch; j++){
            double totalLoss = 0;
            Collections.shuffle(this.trainingSet, this.random);

            for(Instance instance: this.trainingSet) {
                runNeuralNet(instance);
                for(int i = 0; i < this.outputNodes.size(); i++) {
                    Node outNode = this.outputNodes.get(i);
                    outNode.calculateDelta(instance.classValues.get(i), this.outputNodes,i);
                }
                for(int i = 0; i < this.hiddenNodes.size(); i++) {
                    Node hidNode = this.hiddenNodes.get(i);
                    hidNode.calculateDelta(-99999, this.outputNodes, i);
                }
                for(Node node : this.outputNodes) {
                    node.updateWeight(this.learningRate);
                }
                for(Node node : this.hiddenNodes) {
                    node.updateWeight(this.learningRate);
                }
            }

            for(Instance inst : this.trainingSet) {
                totalLoss += loss(inst);
            }

            totalLoss /= this.trainingSet.size();
            System.out.print("Epoch: " + j + ", Loss: ");
            System.out.format("%.3e", totalLoss);
            System.out.println();
        }
    }

    /**
     * Calculate the cross entropy loss from the neural network for
     * a single instance.
     * The parameter is a single instance
     */
    private double loss(Instance instance) {
        runNeuralNet(instance);
        double crossEntropy = 0;
        for(int i =0 ; i < this.outputNodes.size();i++) {
            double g = this.outputNodes.get(i).getOutput();
            crossEntropy -= instance.classValues.get(i) * Math.log(g);
        }
        return crossEntropy;
    }

    private void runNeuralNet(Instance instance) {
        // Set input values on the input nodes
        for(int i = 0; i < this.inputNodes.size() - 1; i++) {
            Node node = this.inputNodes.get(i);
            node.setInput(instance.attributes.get(i));
        }


        for(Node node: this.hiddenNodes) {
            node.getWeightedInputValue();
            node.calculateOutput(null);
        }
        for(Node node: this.outputNodes) {
            node.getWeightedInputValue();
        }
        for(Node node: this.outputNodes) {
            node.calculateOutput(this.outputNodes);
        }
    }
}
