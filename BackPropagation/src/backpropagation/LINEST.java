
package backpropagation;

import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import org.apache.poi.ss.usermodel.Cell;
import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.xssf.usermodel.XSSFSheet;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

public class LINEST {
    private ArrayList<ArrayList<Double>> data = BackPropagation.testingData;
    private double[] weights = new double[data.get(0).size()];
    private double[] predictedOutputs = new double[data.size()];
    private double[] actualOutputs = new double[data.size()];
    private double mse=0;
    
    public LINEST(){
        this.weights[0]=-0.45583798;
        this.weights[1]=0.380802074;
        this.weights[2]=0.405399896;
        this.weights[3]=0.311213449;
        this.weights[4]=0.155398408;
    }
    
    public void run(){
        double total=0;
        for (int i=0; i<data.size(); i++) {
            double[] inputs = new double[data.get(0).size()-1];
            for (int j = 0; j < inputs.length; j++) {
                inputs[j]=data.get(i).get(j);
            }
            actualOutputs[i]=deStandardiseOutput(data.get(i).get(data.get(i).size()-1));
            predictedOutputs[i]=deStandardiseOutput(calculate(inputs));
        }
        for (int i = 0; i < data.size(); i++) {
            total+=Math.pow((predictedOutputs[i]-actualOutputs[i]),2);
        }
        mse=total/(data.size()-1);
        plotPredVsActual();
    }
    
    public void plotPredVsActual(){
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
        try (FileOutputStream outputStream = new FileOutputStream("PredVsActual.xlsx")) {
            workbook.write(outputStream);
        }catch (IOException e) {
            e.printStackTrace();
          }
    }
    
    public double deStandardiseOutput(double value){
        return (((value-0.1)/0.8)*(1.28-0.07)+0.07);
    }
    
    public void printMSE(){
        System.out.println("MSE: "+mse);
    }
    
    public double calculate(double[] inputs){
        double dp = 0;
        for (int i = 0; i < inputs.length; i++) {
            dp+=inputs[i]*weights[i];
        }
        dp+=weights[weights.length-1]; //plus the bias term
        return dp;
    }
}

