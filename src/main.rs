pub mod datamanager;
extern crate nalgebra as na;
use rand::prelude::*;
use rand_distr::{Distribution, Normal, NormalError};

fn coeff_wise_product(v1 : &na::DVector<f64>, v2 : &na::DVector<f64>) -> na::DVector<f64>{
    let mut out = na::DVector::<f64>::zeros(v1.len());
    for i in 0..v1.len(){
        out[i] = v1[i]*v2[i];
    }
    return out;
}

fn relu(num : f64) -> f64{
    if num > 0f64{
        num
    }
    else { 0f64}
}

fn relu_vec(vec : &na::DVector<f64>) -> na::DVector<f64>{
    let mut out = na::DVector::<f64>::zeros(vec.len());
    for i in 0..out.len(){
        out[i] = relu(vec[i]);
    }
    out
}

struct Layer {
    weights : na::DMatrix<f64>,
    biases : na::DVector<f64>,
}

impl Layer {
    pub fn pass_through(&self,input : &na::DVector<f64>) -> na::DVector<f64> {
        &self.weights*input + &self.biases
    }
    
    pub fn initialize_weights(&mut self){
        let normal_distribution = Normal::new(0.0f64,1.0f64).unwrap();
        let mut rng = rand::thread_rng();
        for i in self.weights.iter_mut() {
            *i = normal_distribution.sample(&mut rng);
        } 
        for i in self.biases.iter_mut() {
            *i = normal_distribution.sample(&mut rng);
        }
    }   
}

struct NN {
    layers : Vec<Layer>,
    eta : f64,
    lambda :f64,
    input_size : usize,
    output_size : usize,
}

impl NN {
    pub fn initialize_layers(&mut self, structure : &[usize]){
        self.input_size = structure[0];
        self.output_size = structure[structure.len()-1];
        for i in 1..structure.len(){
            self.layers.push( Layer { weights : na::DMatrix::identity(structure[i], structure[i-1]),
                                	  biases : na::DVector::zeros(structure[i])});
            self.layers[i-1].initialize_weights();
        }        
    }

    pub fn back_prop(&self, input : &na::DVector<f64>,output : &na::DVector<f64>,nabla_w : &mut na::DMatrix<f64>, nabla_b : &mut na::DMatrix<f64>){
        let mut activations : Vec<na::DVector<f64>> = vec![];
        let mut zs : Vec<na::DVector<f64>> = vec![];
        let mut z = self.layers[0].pass_through(input);
        let mut a = relu_vec(&z);
        zs.push(z);
        for i in 1..self.layers.len(){
            z = self.layers[i].pass_through(&a);
            activations.push(a);
            a = relu_vec(&z);
            zs.push(z);
        }

        let mut delta = 

        
    }
}
fn main() {
    let mut a = datamanager::AllData {                                                      
        training_data : vec![],
        test_data: vec![],
    };
    a.read_data();
}
