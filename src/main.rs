pub mod datamanager;
extern crate nalgebra as na;
use std::f64::INFINITY; 
use rand::thread_rng;
use rand::seq::SliceRandom;
use std::env;
use datamanager::Wine;
use na::max;
use rand::prelude::*;
use rand_distr::{Distribution, Normal, NormalError};
use std::num;

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

fn derivative(vec : &na::DVector<f64>)-> na::DVector<f64> {
    let mut out = na::DVector::<f64>::zeros(vec.len());
    for i in 0..out.len(){
        if vec[i]>0.0f64{out[i]=1.0f64;}
        else{out[i]=0.0f64;}
    }
    out
}

fn max_position(vec : &na::DVector<f64>) -> usize {
    let mut max = -INFINITY;
    let mut out: usize= 0;
    for i in 1..vec.len(){
        if vec[i]>max{
            max =vec[i];
            out = i;
        }
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
        let num_cols = self.weights.ncols();
        let mut rng = rand::thread_rng();
        for i in self.weights.iter_mut() {
            *i = normal_distribution.sample(&mut rng)/f64::sqrt(num_cols as f64);
        } 
        for i in self.biases.iter_mut() {
            *i = normal_distribution.sample(&mut rng);
        }
    }   
}

struct NN {
    layers : Vec<Layer>,
    n_size : usize,
    eta : f64,
    lambda :f64,
    input_size : usize,
    output_size : usize,
}

impl NN {
    pub fn initialize_layers(&mut self, structure : &[usize]){
        self.input_size = structure[0];
        self.output_size = structure[structure.len()-1];
        self.n_size = structure.len();
        for i in 1..structure.len(){
            self.layers.push( Layer { weights : na::DMatrix::identity(structure[i], structure[i-1]),
                                	  biases : na::DVector::zeros(structure[i])});
            self.layers[i-1].initialize_weights();
        }        
    }

    pub fn back_prop(&self, input : &na::DVector<f64>,output : &na::DVector<f64>, nabla_w : &mut Vec<na::DMatrix<f64>>, nabla_b : &mut Vec<na::DVector<f64>>){
        let mut activations : Vec<na::DVector<f64>> = vec![];
        let mut zs : Vec<na::DVector<f64>> = vec![];
        let mut z = self.layers[0].pass_through(input);
        let mut a = relu_vec(&z);
        activations.push(na::DVector::<f64>::zeros(11));
        zs.push(z);
        for i in 1..self.layers.len(){
            z = self.layers[i].pass_through(&a);
            activations.push(a);
            a = relu_vec(&z);
            zs.push(z);
        }
        activations.push(a);
        let mut delta = activations.last().unwrap() - output;
        
        nabla_b[self.n_size-2] += &delta;
        nabla_w[self.n_size-2] += &delta * (&activations[activations.len()-2]).transpose();

        for i in 2..self.n_size{
            delta = coeff_wise_product(&((&self.layers[self.n_size-i].weights.transpose()) * &delta), &derivative(&zs[self.n_size-1-i]));
            nabla_b[self.n_size-1-i] += &delta;
            if i == self.n_size {nabla_w[self.n_size-1-i] += &delta * input.transpose();}
            else {nabla_w[self.n_size-1-i] += &delta * (&activations[self.n_size-i-1]).transpose();}
        }
    }

    pub fn get_nabla_b(&self) -> Vec<na::DVector<f64>>{
        let mut out = vec![];

        for layer in self.layers.iter(){
            out.push(na::DVector::<f64>::zeros(layer.biases.len()));
        }
        out
    }

    pub fn get_nabla_w(&self) -> Vec<na::DMatrix<f64>>{
        let mut out = vec![];
        for layer in self.layers.iter() {
            out.push(na::DMatrix::<f64>::zeros(layer.weights.nrows(),layer.weights.ncols()));
        }
        out
    }

    pub fn update_minibatch(&mut self,batch : &Vec<&Wine>,train_data_size : f64){
        let mut nabla_b = self.get_nabla_b();
        let mut nabla_w = self.get_nabla_w();

        for wine in batch.iter() {
            self.back_prop(&wine.features, &wine.label, &mut nabla_w, &mut nabla_b);
        }

        for i in 0..self.n_size-1{
            let batch_size = batch.len() as f64;
            let new_weights = (1.0f64-self.eta*(self.lambda/train_data_size))*(&self.layers[i].weights) - (self.eta / batch_size) * &nabla_w[i];
            self.layers[i].weights = new_weights;
            let minus = (self.eta / batch_size) * (&nabla_b[i]);
            self.layers[i].biases -= minus;
        }
    }

    pub fn feed_forward(&self, input : &na::DVector<f64>) -> na::DVector<f64>{
        let mut out = relu_vec(&self.layers[0].pass_through(input)); 
        for i in 1..self.n_size-1{
            out = relu_vec(&self.layers[i].pass_through(&out));
        }
        out
    }

    pub fn evaulate(&self, test_data : &Vec<Wine>) -> f64 {
        let mut score: f64 = 0.0;
        for wine in test_data.iter(){
            let out = self.feed_forward(&wine.features);
            let pos = max_position(&out);
            let correct = max_position(&wine.label);
            //print!("{},{}",out,wine.label);
            if pos == correct {score += 1.0}
        }
        score
    }

    pub fn sgd(&mut self, mut training_data : Vec<Wine>,test_data : Vec<Wine>, epochs : i32,batch_size : usize){
        let l = training_data.len();
        let t = test_data.len();

        for j in 0..epochs{
            training_data.shuffle(&mut thread_rng());
            let mut batches : Vec<Vec<&Wine>> = vec![];
            let mut total_index : usize = 0;
            for k in 0..(t/batch_size){
                let mut batch :Vec<&Wine> = vec![];
                for i in 0..batch_size{
                    batch.push(&training_data[total_index]);
                    total_index +=1;
                }
                batches.push(batch);
            }
        
            for batch in batches.iter(){
                self.update_minibatch(batch, l as f64);
            }
            let score = self.evaulate(&test_data);
            let size = t as f64;
            let performance = score / size;
            print!("performance on epoch {} :  {}",j,performance);
            print!("\n");
        }
    }

}
fn main() {
    //env::set_var("RUST_BACKTRACE", "full");
    let mut a = datamanager::AllData {                                                      
        training_data : vec![],
        test_data: vec![],
    };

    let mut network = NN {eta : 0.001, input_size : 11, lambda : 0.1, layers : vec![],n_size : 0, output_size : 10};
    a.read_data();
    NN::initialize_layers(&mut network, &[11,100,100,10]);
    NN::sgd(&mut network, a.training_data,a.test_data,2000,3);
}
