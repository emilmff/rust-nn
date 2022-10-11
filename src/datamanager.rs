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
        let dir : String = String::from("C:/Users/emil9/Downloads/winequality-white.csv");
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
                    println!("shits fucked yo");
                }
                if wine_counter < 4100{self.training_data.push(wine)}
                else{self.test_data.push(wine)}
            }
            wine_counter +=1;
        }
    }
}