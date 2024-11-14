
package backpropagation;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.atomic.AtomicLong;
import org.apache.poi.openxml4j.exceptions.InvalidFormatException;
import org.apache.poi.ss.usermodel.Cell;
import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.ss.usermodel.Sheet;
import org.apache.poi.ss.usermodel.Workbook;
import org.apache.poi.xssf.usermodel.XSSFCell;
import org.apache.poi.xssf.usermodel.XSSFRow;
import org.apache.poi.xssf.usermodel.XSSFSheet;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

public class Basic {
    protected final ArrayList<ArrayList<Double>> trainingData = BackPropagation.trainingData;
    protected final ArrayList<ArrayList<Double>> validationData = BackPropagation.validationData;
    protected final ArrayList<ArrayList<Double>> testingData = BackPropagation.testingData;
    protected double[][] weights;
    protected double[] biases;
    protected int noInput;
    protected int noHidden;
    protected int noEpochs;
    protected double p; //Learning Parameter
    protected double mse;
    
    public Basic(int noInput, int noHidden, int noEpochs, double p){
        this.weights=new double[noInput+1][noHidden];
        this.biases=new double[noHidden+1];
        this.noInput=noInput;
        this.noHidden=noHidden;
        this.noEpochs=noEpochs;
        this.p=p;
        this.mse=0;
        this.createNetwork();
    }
    
    public Basic(int noInput, int noHidden, int noEpochs){
        this.weights=new double[noInput+1][noHidden];
        this.biases=new double[noHidden+1];
        this.noInput=noInput;
        this.noHidden=noHidden;
        this.noEpochs=noEpochs;
        this.mse=0;
        this.createNetwork();
    }
    
    public void createNetwork(){
        Random seedGenerator = new Random(); // create a Random object to generate seed
        long seed = seedGenerator.nextLong(); // generate a long seed
        //create the weights from the input to hidden nodes
        Random rnd = new Random(-1303657781631004123L);
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
        System.out.println("Seed value: " + seed);
    }
    
    public void run(){
        for (int i = 0; i < noEpochs+1; i++) {
            if(i % 500 != 0){ //Training cycle
                training();
                //printWeights(weights,biases);
            }else{ //Validation cycleD
                validation();
                plotTrainingVsTesting(i);
            }
        }
        testing();
        System.out.println("Train MSE: " + getMSE(trainingData));
        printWeights();
    }
    
    
    public void training(){
        for (int j = 0; j < trainingData.size(); j++) {
            double[] u = new double[noInput + noHidden + 1]; //create input set
            double[] deltas = new double[noHidden + 1];
            for (int l = 0; l < noInput; l++) {
                u[l]=trainingData.get(j).get(l); //stores all the starting values in u list
            }
            double  output=trainingData.get(j).get(trainingData.get(j).size()-1); //gets the output value at the end of the list
            forwardPass(u);
            backwardPass(u,deltas,output);
        }
    }
    
    public double getMSE(ArrayList<ArrayList<Double>> data){
        double[] predictedOutputs = new double[data.size()];
        double[] actualOutputs = new double[data.size()];
        //initialize u inputs and outputs
        for (int j = 0; j < data.size(); j++) {
            double[] u = new double[noInput+noHidden+1]; //creates new u object
            for (int k = 0; k < noInput; k++) {
                u[k] = data.get(j).get(k); //initializes all of the input u values
            }
            predictedOutputs[j] = deStandardiseOutput(forwardPass(u)); //set the predicted output to whatever final u value is returned on the forward pass
            actualOutputs[j]= deStandardiseOutput(data.get(j).get((data.get(j).size())-1));
        }
        //printArray(actualOutputs);
        double total = 0;
        //Mean Squared Error calculation
        for (int j = 0; j < data.size(); j++) {
            total += Math.pow((predictedOutputs[j]-actualOutputs[j]),2); //(predicted value - actual value)^2
        }
        return (total/data.size());
    }
    
    public void validation(){
        double[] predictedOutputs = new double[validationData.size()];
        double[] actualOutputs = new double[validationData.size()];
        //initialize u inputs and outputs
        for (int j = 0; j < validationData.size(); j++) {
            double[] u = new double[noInput+noHidden+1]; //creates new u object
            for (int k = 0; k < noInput; k++) {
                u[k] = validationData.get(j).get(k); //initializes all of the input u values
            }
            predictedOutputs[j] = deStandardiseOutput(forwardPass(u)); //set the predicted output to whatever final u value is returned on the forward pass
            actualOutputs[j]=deStandardiseOutput(validationData.get(j).get(validationData.get(j).size()-1));
        }
        double total = 0;
        //Mean Squared Error calculation
        for (int j = 0; j < validationData.size(); j++) {
            total += Math.pow((predictedOutputs[j]-actualOutputs[j]),2); //(predicted value - actual value)^2
        }
        mse = total/validationData.size(); //update MSE
        System.out.println("Validation MSE: " + mse);
    }
    
    public void testing(){
        double[] predictedOutputs = new double[testingData.size()];
        double[] actualOutputs = new double[testingData.size()];
        for (int j = 0; j < testingData.size(); j++) {
            double[] u = new double[noInput+noHidden+1]; //creates new u object
            for (int k = 0; k < noInput; k++) {
                u[k] = testingData.get(j).get(k); //initializes all of the input u values
            }
            predictedOutputs[j] = deStandardiseOutput(forwardPass(u)); //set the predicted output to whatever final u value is returned on the forward pass
            actualOutputs[j]=deStandardiseOutput(testingData.get(j).get(testingData.get(j).size()-1));
        }
        double total = 0;
        //Mean Squared Error calculation
        for (int j = 0; j < testingData.size(); j++) {
            total += Math.pow((predictedOutputs[j]-actualOutputs[j]),2); //(predicted value - actual value)^2
        }
        mse = total/testingData.size(); //update MSE
        //System.out.println("MSE: " + mse);
        System.out.println("Testing MSE: " +(total/testingData.size()));
        plotPredVsActual(testingData,predictedOutputs,actualOutputs);
    }
    
    public double deStandardiseOutput(double value){
        return (((value-0.1)/0.8)*(1.28-0.07)+0.07);
    }
  
    
    public void printMSE(){
        System.out.println("MSE " + mse);;
    }
    
    
    public double forwardPass(double[] u){
        //forward pass for the weights from the input to hidden nodes
        double S = 0;
        for(int i=0; i<noHidden; i++){
            S=biases[i];
            for (int j = 0; j < noInput ; j++) {
                S += weights[j][i]*u[j];
            }
            u[i+noInput] = sigmoid(S);
        }
        
        //pass from the hidden nodes to the output node
        S=biases[biases.length-1];
        for (int i = 0; i < noHidden; i++) {
            S += weights[weights.length-1][i]*u[noInput+i];
        }
        u[u.length-1] = sigmoid(S);
        
        return u[u.length-1]; //returns the final predicted output for that row
    }
    
    
    
    
    public void backwardPass(double[] u, double[] deltas, double output){
        double deriv = u[u.length-1]*(1-u[u.length-1]); //gets the derivative of the final output where u[noInput+noHidden] is the last element of u
        deltas[deltas.length-1] = (output - u[u.length-1])*deriv; //get last delta value
        //Construct the rest of the deltas using this
        for (int i = 0; i < noHidden; i++) {
            double newDeriv = u[noInput+i]*(1-u[noInput+i]);
            deltas[i] = weights[weights.length-1][i]*deltas[deltas.length-1]*newDeriv;
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
            weights[weights.length-1][i] += (p*deltas[deltas.length-1]*u[i+noInput]);
        }
        
        //update bias for output node
        biases[biases.length-1] += (p*deltas[deltas.length-1]);
    }
    
    public void printWeights(){
        System.out.println("Weights:");
        System.out.println("["+Arrays.deepToString(weights).replace("], ", "]\n").replace("[[", "[").replace("]]", "]")+"]");
        System.out.println();
        System.out.println("Biases:");
        System.out.println(Arrays.toString(biases));
    }
    
    public void plotTrainingVsTesting(int i){
        double trainingMSE = getMSE(trainingData);
        double testingMSE = getMSE(testingData);
        try {
            FileInputStream fis = new FileInputStream(new File("ErrorPlot.xlsx")); //Load excel file
            XSSFWorkbook workbook = new XSSFWorkbook(fis);
            
            XSSFSheet sheet = workbook.getSheetAt(0);
            
            int lastRowNum = sheet.getLastRowNum(); //get last row number
            
            XSSFRow newRow = sheet.createRow(lastRowNum + 1); //go to next row
            
            //Create values for each cell
            XSSFCell cell1 = newRow.createCell(0);
            cell1.setCellValue(i);
            XSSFCell cell2 = newRow.createCell(1);
            cell2.setCellValue(trainingMSE);
            XSSFCell cell3 = newRow.createCell(2);
            cell3.setCellValue(testingMSE);
            
            //Save changes
            FileOutputStream fos = new FileOutputStream("ErrorPlot.xlsx");
            workbook.write(fos);
            fos.close();
            
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    
    
    public void plotPredVsActual(ArrayList<ArrayList<Double>> data, double[] predictedOutputs, double[] actualOutputs){
        //plot data
        double[][] predVsActual = new double[data.size()][2];
        for (int i = 0; i < data.size(); i++) {
            predVsActual[i][0] = predictedOutputs[i];
            predVsActual[i][1] = actualOutputs[i];
        }
        XSSFWorkbook workbook = new XSSFWorkbook();
        XSSFSheet sheet = workbook.createSheet("Data");

        int rowNum = 0;
        for (double[] rowData : predVsActual) {
          Row row = sheet.createRow(rowNum++);

          int colNum = 0;
          for (double field : rowData) {
            Cell cell = row.createCell(colNum++);
            cell.setCellValue(field);
          }
        }
        System.out.println("testing");
        try (FileOutputStream outputStream = new FileOutputStream("PredVsActual.xlsx")) {
            workbook.write(outputStream);
        }catch (IOException e) {
            e.printStackTrace();
          }
    }
    
    
    public double sigmoid(double S){
        return 1/(1+Math.exp(-1 *S));
    }
}
