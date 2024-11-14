
package backpropagation;

import java.util.Random;

public class Momentum extends Basic{
        
    private double[][] prevWeights;
    private double[] prevBiases;
    private double a;

    public Momentum(int noInput, int noHidden, int noEpochs, double p, double a){
        super(noInput,noHidden,noEpochs,p);
        this.a=a;
    }

    
    @Override
    public void createNetwork(){
        this.prevWeights=new double[noInput+1][noHidden];
        this.prevBiases=new double[noHidden+1];
        //create the weights from the input to hidden nodes
        Random seedGenerator = new Random(); //create a Random object to generate seed
        long seed = seedGenerator.nextLong(); //generate a long seed
        //create the weights from the input to hidden nodes
        Random rnd = new Random(seed);
        for (int i = 0; i < noInput; i++) {
            for (int j = 0; j < noHidden; j++) {
                this.weights[i][j] = rnd.nextDouble() * (4.0 / noInput) - (2.0 / noInput);
                this.prevWeights[i][j] = weights[i][j];
            }
        }

        //create the weights from hidden to output node
        for (int j = 0; j < noHidden; j++) {
            this.weights[noInput][j] = rnd.nextDouble() * (4.0 / noInput) - (2.0 / noInput);
            this.prevWeights[noInput][j] = weights[noInput][j];
        }

        //create the biases for the hiddens and output node
        for (int i = 0; i < noHidden + 1; i++) {
            this.biases[i] = rnd.nextDouble() * (4.0 / noInput) - (2.0 / noInput); //random value between [-2/n,2/n]
            this.prevBiases[i]=biases[i];
        }
        System.out.println("Seed value: "+seed);
        
    }

    @Override
    public void backwardPass(double[] u, double[] deltas, double output){
        double deriv = u[u.length-1]*(1-u[u.length-1]); //gets the derivative of the final output where u[noInput+noHidden] is the last element of u
        deltas[deltas.length-1] = (output - u[u.length-1])*deriv; //get last delta value

        double change;
        //Construct the rest of the deltas using this
        for (int i = 0; i < noHidden; i++) {
            double newDeriv = u[noInput+i]*(1-u[noInput+i]);
            deltas[i] = weights[noInput][i]*deltas[deltas.length-1]*newDeriv;
        }
        //update weights from input to hidden
        for (int i = 0; i < noInput; i++) {
            for (int j = 0; j < noHidden; j++) {
                change = weights[i][j] - prevWeights[i][j];
                prevWeights[i][j] = weights[i][j];
                weights[i][j] += (p*deltas[j]*u[i]) + (a*change);
            }
        }
        //update biases for hidden layer
        for (int i = 0; i < noHidden; i++) {
            change = biases[i] - prevBiases[i];
            prevBiases[i] = biases[i];
            biases[i] += (p*deltas[i]) + (a*change);
        }

        //update weights from hidden to output
        for (int i = 0; i < noHidden; i++) {
            change = weights[noInput][i] - prevWeights[noInput][i];
            prevWeights[noInput][i] = weights[noInput][i];
            weights[noInput][i] += (p*deltas[deltas.length-1]*u[i+noInput]) + (a*change);
            //should be noInput I think?
        }

        //update bias for output node
        change = biases[biases.length-1] - prevBiases[prevBiases.length-1];
        prevBiases[prevBiases.length-1] = biases[biases.length-1];
        biases[biases.length-1] += (p*deltas[deltas.length-1]) + (a*change);
    }
    
    
    
}

