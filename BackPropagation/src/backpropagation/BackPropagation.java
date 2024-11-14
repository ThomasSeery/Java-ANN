package backpropagation;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.Scanner;
import org.apache.poi.ss.usermodel.Cell;
import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.xssf.usermodel.XSSFSheet;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;


public class BackPropagation {
    
    public static ArrayList<ArrayList<Double>> trainingData = new ArrayList<>();
    public static ArrayList<ArrayList<Double>> validationData = new ArrayList<>();
    public static ArrayList<ArrayList<Double>> testingData = new ArrayList<>();
    
    
    public static void main(String[] args) {
        
        String[] dataSet = {"TrainingData.xlsx","ValidationData.xlsx","TestingData.xlsx"};
        String[] dataSetWoDSP = {"TrainingDataWoDSP.xlsx","ValidationDataWoDSP.xlsx","TestingDataWoDSP.xlsx"};
        
        try {
            // read from first data set
            FileInputStream fis1 = new FileInputStream(new File(dataSet[0]));  
            XSSFWorkbook wb1 = new XSSFWorkbook(fis1);   
            XSSFSheet sheet1 = wb1.getSheetAt(0);   
            for (Row row : sheet1) {
                ArrayList<Double> rowData = new ArrayList<>();
                // start from the second cell
                for (Cell cell: row) {
                    if (cell.getCellType() == Cell.CELL_TYPE_NUMERIC) {
                        rowData.add(cell.getNumericCellValue());   
                    }
                }
                trainingData.add(rowData);
            }  

            // read from second data set
            FileInputStream fis2 = new FileInputStream(new File(dataSet[1]));  
            XSSFWorkbook wb2 = new XSSFWorkbook(fis2);   
            XSSFSheet sheet2 = wb2.getSheetAt(0);   
            for (Row row : sheet2) {
                ArrayList<Double> rowData = new ArrayList<>();
                // start from the second cell
                for (Cell cell: row) {
                    if (cell.getCellType() == Cell.CELL_TYPE_NUMERIC) {
                        rowData.add(cell.getNumericCellValue());   
                    }
                }
                validationData.add(rowData);
            }
            
            // read from third data set
            FileInputStream fis3 = new FileInputStream(new File(dataSet[2]));  
            XSSFWorkbook wb3 = new XSSFWorkbook(fis3);   
            XSSFSheet sheet3 = wb3.getSheetAt(0);   
            for (Row row : sheet3) {
                ArrayList<Double> rowData = new ArrayList<>();
                // start from the second cell
                for (Cell cell: row) {
                    if (cell.getCellType() == Cell.CELL_TYPE_NUMERIC) {
                        rowData.add(cell.getNumericCellValue());   
                    }
                }
                testingData.add(rowData);
            }

        } catch(IOException e) {
            e.printStackTrace();
        }

        
        Basic b = new Basic(5,3,63500,0.1);
        b.run(); 
        
        System.out.println("---------------------------------");
        
        BoldDriver bd = new BoldDriver(4,8,24500,new LearningParameter(0.1,0.01,0.5,0.05,0.3));
        //bd.run(); //uncomment this to run this model
        
        System.out.println("---------------------------------");
        
        Momentum m = new Momentum(4,8,5000,0.1,0.9);
        //m.run(); //uncomment this to run this model
        
        System.out.println("---------------------------------");
        
        Annealing a = new Annealing(4,8,38500,0.1,0.01);
        //a.run(); 
        
        System.out.println("---------------------------------");
        
        WeightDecay wd = new WeightDecay(4,8,100000,0.1);
        //wd.run(); //uncomment this to run this model
        
        System.out.println("---------------------------------");
        
        LINEST linest = new LINEST();
        //linest.run(); //uncomment this to run this model
        //linest.printMSE();
    }
}
