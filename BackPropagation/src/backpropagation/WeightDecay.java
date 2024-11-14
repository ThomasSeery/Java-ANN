package backpropagation;

public class WeightDecay extends Basic{

    public WeightDecay(int noInput, int noHidden, int noEpochs, double p) {
        super(noInput, noHidden, noEpochs, p);
    }

    @Override
    public void run(){
        for (int i = 0; i < noEpochs+1; i++) {
            if(i % 500 != 0){ //Training cycle
                training(i);
            }else{ //Validation cycle
                validation();
                plotTrainingVsTesting(i);
            }
        }
        testing();
        printWeights();
    }
    
    public void training(int i){
        for (int j = 0; j < trainingData.size(); j++) {
            double[] u = new double[noInput + noHidden + 1]; //create input set
            double[] deltas = new double[noHidden + 1];
            for (int l = 0; l < noInput; l++) {
                u[l]=trainingData.get(j).get(l); //stores all the starting values in u list
            }
            double output=trainingData.get(j).get(trainingData.get(j).size()-1); //gets the output value at the end of the list
            forwardPass(u);
            backwardPass(u,deltas,output,i);
        }
    }

    public void backwardPass(double[] u, double[] deltas, double output, int x){
        double deriv = u[u.length-1]*(1-u[u.length-1]); //gets the derivative of the final output where u[noInput+noHidden] is the last element of u
        deltas[deltas.length-1] = (output - u[u.length-1] + (upsilon(x)*omega()))*deriv; //get last delta value
        //Construct the rest of the deltas using this
        for (int i = 0; i < noHidden; i++) {
            double newDeriv = u[noInput+i]*(1-u[noInput+i]);
            deltas[i] = weights[noInput][i]*deltas[deltas.length-1]*newDeriv;
        }
        //update weights from input to hidden
        for (int i = 0; i < noInput; i++) {
            for (int j = 0; j < noHidden; j++) {
                weights[i][j] += (p*deltas[j]*u[i]);
            }
        }
        //update biases for hidden layer
        for (int i = 0; i < noHidden; i++) {
            biases[i] += (p*deltas[i]);
        }

        //update weights from hidden to output
        for (int i = 0; i < noHidden; i++) {
            weights[noInput][i] += (p*deltas[deltas.length-1]*u[i+noInput]);
        }

        //update bias for output node
        biases[biases.length-1] += (p*deltas[deltas.length-1]);
    }

    public double omega(){
        double o = 0;
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[0].length; j++) {
                o+=Math.pow(weights[i][j],2);
            }
        }
        for (int i = 0; i < biases.length; i++) {
            o+=Math.pow(biases[i],2);
        }
        o/=(2*((weights.length*weights[0].length)+biases.length));
        return o;
    }

    public double upsilon(int x){
        return 1/(p*x);
    }

}


