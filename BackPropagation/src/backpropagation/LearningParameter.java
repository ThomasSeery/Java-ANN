/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package backpropagation;

public class LearningParameter {
    private double value;
    private double min;
    private double max;
    private double inc;
    private double dec;
    
    public LearningParameter(double value,double min, double max, double inc,double dec){
        this.value=value;
        this.min=min;
        this.max=max;
        this.inc=1+inc;
        this.dec=1-dec;
    }
    
    public double getValue(){
        return value;
    }
    
    public void increase(){
        if((this.value*inc)>=max){
            this.value=max;
        }else{
            this.value*=inc;
        }
    }
    
    public void decrease(){
        if((this.value*dec)<=min){
            this.value=min;
        }else{
            this.value*=dec;
        }
    }
    
    
}
