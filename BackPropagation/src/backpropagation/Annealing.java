
package backpropagation;

public class Annealing extends Basic{

    private double ep;

    public Annealing(int noInput, int noHidden, int noEpochs, double p, double ep) {
        super(noInput, noHidden, noEpochs, p);
        this.ep=ep;
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
        deltas[deltas.length-1] = (output - u[u.length-1])*deriv; //get last delta value
        //Construct the rest of the deltas using this
        for (int i = 0; i < noHidden; i++) {
            double newDeriv = u[noInput+i]*(1-u[noInput+i]);
            deltas[i] = weights[noInput][i]*deltas[deltas.length-1]*newDeriv;
        }
        //update weights from input to hidden
        for (int i = 0; i < noInput; i++) {
            for (int j = 0; j < noHidden; j++) {
                weights[i][j] += (p(x)*deltas[j]*u[i]);
            }
        }
        //update biases for hidden layer
        for (int i = 0; i < noHidden; i++) {
            biases[i] += (p(x)*deltas[i]);
        }

        //update weights from hidden to output
        for (int i = 0; i < noHidden; i++) {
            weights[noInput][i] += (p(x)*deltas[deltas.length-1]*u[i+noInput]);
        }

        //update bias for output node
        biases[biases.length-1] += (p(x)*deltas[deltas.length-1]);
    }

    public double p(int x){
        return ep+(p-ep)*(1-(1/(1+Math.exp(10-((20*(x))/noEpochs))))); //returns learning parameter based on current number of epochs
    }

}
