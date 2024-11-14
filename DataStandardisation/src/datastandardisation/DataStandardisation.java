
package datastandardisation;

import java.io.*;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.text.DecimalFormat;
import java.util.*;  
import org.apache.poi.hssf.usermodel.HSSFSheet;
import org.apache.poi.hssf.usermodel.HSSFWorkbook;
import org.apache.poi.ss.usermodel.Cell;  
import org.apache.poi.ss.usermodel.Row;  
import org.apache.poi.xssf.usermodel.XSSFSheet;  
import org.apache.poi.xssf.usermodel.XSSFWorkbook;  

public class DataStandardisation {

    public static void main(String[] args) { 
        ArrayList<ArrayList<Double>> tableData = new ArrayList<>();
        try {
            FileInputStream fis = new FileInputStream(new File("DataSet.xlsx"));  
            XSSFWorkbook wb = new XSSFWorkbook(fis);   
            XSSFSheet sheet = wb.getSheetAt(0);     
            for (Row row : sheet) {
                ArrayList<Double> rowData = new ArrayList<>();
                // start from the second cell
                for (Cell cell: row) {
                    if (cell.getCellType() == Cell.CELL_TYPE_NUMERIC) {
                        rowData.add(cell.getNumericCellValue());   
                    }
                }
                tableData.add(rowData);
            }  
        } catch(IOException e) {
            // handle the exception
        }
        
        System.out.println(tableData.size());
        
        double max[] = {28.9,952.1,743.2,102.8,95,1.28};
        double min[] = {7.2,96.1,78.4,100.2,10,0.07};

        int n = tableData.size(), m = tableData.get(0).size();
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                double newVal = 0.8*((tableData.get(i).get(j)-min[j])/(max[j]-min[j]))+0.1;
                tableData.get(i).set(j,newVal);
            }
        }
        try{
            upload(tableData);
        }catch(Exception e){
            
        }
        


    }
    
    
    public static void upload(ArrayList<ArrayList<Double>> data) throws FileNotFoundException, IOException{
        //create a new workbook and sheet
        XSSFWorkbook workbook = new XSSFWorkbook();
        XSSFSheet sheet = workbook.createSheet("Sheet1");
        
        //iterate over the data and create rows and cells in the sheet
        int rowIndex = 0;
        for (ArrayList<Double> rowData : data) {
            Row row = sheet.createRow(rowIndex++);
            int cellIndex = 0;
            for (double cellData : rowData) {
                Cell cell = row.createCell(cellIndex++);
                cell.setCellValue(cellData);
            }
        }
        
        //write the workbook to an data excel file
        FileOutputStream fos = new FileOutputStream("data.xlsx");
        workbook.write(fos);
        fos.close();
        workbook.close();
    }
    
}

