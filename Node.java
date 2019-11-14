import java.util.*;

/**
 * Class for internal organization of a Neural Network.
 * There are 5 types of nodes. Check the type attribute of the node for details.
 * Feel free to modify the provided function signatures to fit your own implementation
 */

class Node {
    private int type = 0; //0=input,1=biasToHidden,2=hidden,3=biasToOutput,4=Output
    //Array List that will contain the parents (including the bias node) with weights if applicable
    ArrayList<NodeWeightPair> parents = null;
    private double inputValue = 0.0;
    private double outputValue = 0.0;
    private double delta = 0.0; //input gradient

    //Create a node with a specific type
    Node(int type) {
        if (type > 4 || type < 0) {
            System.out.println("Incorrect value for node type");
            System.exit(1);

        } else {
            this.type = type;
        }

        if (type == 2 || type == 4) {
            parents = new ArrayList<>();
        }
    }

    // For an input node sets the input value which will be the value of a particular attribute
    void setInput(double inputValue) {
        if (type != 1 && type != 3) {
            this.inputValue = inputValue;
        }
    }

    /**
     * Calculate the output of a node.
     * You can get this value by using getOutput()
     */
    void calculateOutput(ArrayList<Node> outputNodes) {
        // ReLU for hidden nodes
        if(this.type == Constants.HIDDEN) {
            this.outputValue = Math.max(0, this.inputValue);
        }
        // SoftMax for output nodes
        else if (this.type == Constants.OUTPUT) {
            this.outputValue = this.softMax(outputNodes);
        }
        // Input and bias node outputs never change
    }

    // Gets the output value
    double getOutput() {
        // Input nodes just return their value
        if (this.type == Constants.INPUT) {
            return this.inputValue;
        // Bias nodes always 1
        } else if (this.type == Constants.HIDDEN_BIAS || this.type == Constants.OUTPUT_BIAS) {
            return 1.00;
        // Hidden and output nodes use activation function value
        } else {
            return this.outputValue;
        }

    }

    private double getDelta() {
        if(this.type == Constants.HIDDEN || this.type == Constants.OUTPUT) {
            return delta;
        }

        return 0;
    }

    private double getWeightedOutput(ArrayList<Node> outputNodes, int nodeIndex) {
        double value = 0;
        for(Node node: outputNodes) {
            value += node.parents.get(nodeIndex).weight * node.getDelta();
        }
        return value;
    }

    // Calculate the delta value of a node.
    void calculateDelta(double targetValue, ArrayList<Node> outputNodes, int nodeIndex) {
        if (this.type == Constants.HIDDEN || this.type == Constants.OUTPUT)  {
            double delta;
            if(this.type == Constants.HIDDEN) {
                delta = this.getGPrimeRelu() * getWeightedOutput(outputNodes,nodeIndex);
            }
            else {
                delta = targetValue - this.outputValue;
            }
            this.delta = delta;
        }
    }


    // Update the weights between parents node and current node
    void updateWeight(double learningRate) {
        if (this.type == Constants.HIDDEN || this.type == Constants.OUTPUT) {
            for(NodeWeightPair parentPair: this.parents) {
                double deltaW = learningRate * parentPair.node.getOutput() * delta;
                parentPair.weight += deltaW;
            }
        }
    }


    private double getGPrimeRelu() {
        if(this.type == Constants.HIDDEN) {
            if(this.inputValue <= 0) {
                return 0;
            }
            else {
                return 1;
            }
        }
        return -1;
    }

    private double softMax(ArrayList<Node> outputNodes) {
        double sum = 0;
        double z = Math.exp(this.inputValue);
        for(Node node : outputNodes) {
            sum += Math.exp(node.inputValue);
        }
        return z / sum;
    }

    /**
     * Take output values from parent nodes, weight them, set inputValue
     */
    void getWeightedInputValue() {
        if(this.type == Constants.HIDDEN || this.type == Constants.OUTPUT) {
            double input = 0;
            for(NodeWeightPair pair:parents) {
                input += pair.node.getOutput() * pair.weight;
            }
            this.inputValue = input;
        }
    }
}