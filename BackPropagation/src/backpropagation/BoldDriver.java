
package backpropagation;

import java.util.Random;

public class BoldDriver extends Basic{

    private LearningParameter p;
    private double[][] weightChanges;
    private double[] biasChanges;
    private Double prevMse;

    public BoldDriver(int noInput, int noHidden, int noEpochs, LearningParameter p) {
        super(noInput, noHidden, noEpochs);
        this.p=p;
        this.prevMse=null;
    }
    
    @Override
    public void createNetwork(){
        this.weightChanges = new double[noInput+1][noHidden];
        this.biasChanges = new double[noHidden+1];
        Random seedGenerator = new Random(); // create a Random object to generate seed
        long seed = seedGenerator.nextLong(); // generate a long seed
        //create the weights from the input to hidden nodes
        Random rnd = new Random(seed);
        for (int i = 0; i < noInput; i++) {
            for (int j = 0; j < noHidden; j++) {
                this.weights[i][j] = rnd.nextDouble() * (4.0 / noInput) - (2.0 / noInput);
            }
        }

        //create the weights from hidden to output node
        for (int j = 0; j < noHidden; j++) {
            this.weights[noInput][j] = rnd.nextDouble() * (4.0 / noInput) - (2.0 / noInput);
        }

        //create the biases for the hiddens and output node
        for (int i = 0; i < noHidden + 1; i++) {
            this.biases[i] = rnd.nextDouble() * (4.0 / noInput) - (2.0 / noInput); //random value between [-2/n,2/n]
        }
        System.out.println("Seed value: "+seed);
    }
    
    public void revert(){
        //set weights to prevWeights
        for (int i = 0; i < weights.length-1; i++) {
            for (int j = 0; j < weights[0].length; j++) {
                weights[i][j]-=weightChanges[i][j];
            }
        }
        //set biases to prevBiases
        for (int i = 0; i < biases.length-1; i++) {
            biases[i]-=biasChanges[i];
        }
    }
    
    @Override
    public void validation(){
        super.validation();
        if(prevMse != null){
            double percentChange = ((mse-prevMse)/prevMse)*100;
            if(percentChange>=0.02){ //error function increase
                revert(); //reverts the weight and bias changes
                p.decrease();
            }else if(percentChange<=0){ //error function decrease
                p.increase();
            }
        }
        prevMse = mse;
    }

    
    @Override
    public void backwardPass(double[] u, double[] deltas, double output){
        double deriv = u[u.length-1]*(1-u[u.length-1]); //gets the derivative of the final output where u[noInput+noHidden] is the last element of u
        deltas[deltas.length-1] = (output - u[u.length-1])*deriv; //get last delta value
        //Construct the rest of the deltas using this
        for (int i = 0; i < noHidden; i++) {
            double newDeriv = u[noInput+i]*(1-u[noInput+i]);
            deltas[i] = weights[noInput][i]*deltas[deltas.length-1]*newDeriv;
        }
        //update weights from input to hidden
        for (int i = 0; i < noInput; i++) {
            for (int j = 0; j < noHidden; j++) {
                weights[i][j] += (p.getValue()*deltas[j]*u[i]);
                weightChanges[i][j] = (p.getValue()*deltas[j]*u[i]); //update the weight changes
            }
        }
        //update biases for hidden layer
        for (int i = 0; i < noHidden; i++) {
            biases[i] += (p.getValue()*deltas[i]);
            biasChanges[i] = (p.getValue()*deltas[i]); //update the bias changes
        }

        //update weights from hidden to output
        for (int i = 0; i < noHidden; i++) {
            weights[noInput][i] += (p.getValue()*deltas[deltas.length-1]*u[i+noInput]);
            weightChanges[noInput][i] = (p.getValue()*deltas[deltas.length-1]*u[i+noInput]);
        }

        //update bias for output node
        biases[biases.length-1] += (p.getValue()*deltas[deltas.length-1]);
        biasChanges[biasChanges.length-1] = (p.getValue()*deltas[deltas.length-1]);
    }


}
