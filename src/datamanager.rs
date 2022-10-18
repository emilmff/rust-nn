extern crate nalgebra as na;
use std::fs::File;
use std::io::{prelude::*, BufReader};

pub struct Wine {
    pub features : na::DVector<f64>,
    pub label : na::DVector<f64>,
}   
pub struct AllData {
    pub training_data : Vec<Wine>,
    pub test_data : Vec<Wine>,
}

impl AllData{ 
    pub fn read_data(&mut self){
        let dir : String = String::from("C:/Users/emil9/Downloads/winequality-red.csv");
        let file = File::open(dir).expect("didnt open");
        let reader = BufReader::new(file);

        let mut wine_counter = 0;

        for line in reader.lines() {
            if wine_counter > 0 {
                let mut wine= Wine {
                    features : na::DVector::<f64>::zeros(11),
                    label : na::DVector::<f64>::zeros(10),
                };
                if let Ok(l) = line{
                    let mut trait_counter = 0;
                    let mut num = String::from(""); 
                    for i in l.chars(){
                        if i != ';'{
                            num.push(i);
                            if trait_counter == 11{
                                wine.label[(num.to_string()).trim().parse::<usize>().unwrap()] = 1f64;
                            }
                        }
                        else{
                            if trait_counter <11{
                                wine.features[trait_counter] = (num.to_string()).trim().parse::<f64>().unwrap();
                            }
                            trait_counter +=1;
                            num = String::from("");
                        }
                    }
                }
                else{
                    println!(" yo");
                }
                if wine_counter < 1500{self.training_data.push(wine)}
                else{self.test_data.push(wine)}
            }
            wine_counter +=1;
        }
    }

    pub fn normalize_training_data(&mut self) {
        let mut mins: Vec<f64> = vec![];
        let mut maxs: Vec<f64> = vec![];
        let len_data = self.training_data.len();
        let mut v : &na::DVector<f64> = &self.training_data[0].features;
        let len_features = v.nrows();

        for i in 0..v.nrows(){
            mins.push(v[i]);
            maxs.push(v[i]);
        }

        for i in 1..len_data{
            v = &self.training_data[i].features;

            for j in 0..len_features{
                let val = v[j];
                if val < mins[j] {mins[j]= val;}
                if val >maxs[j] {maxs[j] = val;}
            }
        }

        
        for i in 0..len_data{
            for j in 0..len_features{
                if maxs[j]-mins[j] == 0.0 {self.training_data[i].features[j] = 0.0;}
                else {self.training_data[i].features[j] =(self.training_data[i].features[j]-mins[j])/(maxs[j]-mins[j])}
            }
        }
    }
    
    pub fn normalize_test_data(&mut self){
        let mut mins: Vec<f64> = vec![];
        let mut maxs: Vec<f64> = vec![];
        let len_data = self.test_data.len();
        let mut v : &na::DVector<f64> = &self.test_data[0].features;
        let len_features = v.nrows();

        for i in 0..v.nrows(){
            mins.push(v[i]);
            maxs.push(v[i]);
        }

        for i in 1..len_data{
            v = &self.test_data[i].features;

            for j in 0..len_features{
                let val = v[j];
                if val < mins[j] {mins[j]= val;}
                if val >maxs[j] {maxs[j] = val;}
            }
        }

        
        for i in 0..len_data{
            for j in 0..len_features{
                if maxs[j]-mins[j] == 0.0 {self.test_data[i].features[j] = 0.0;}
                else {self.test_data[i].features[j] =(self.test_data[i].features[j]-mins[j])/(maxs[j]-mins[j])}
            }
        }
    }
}