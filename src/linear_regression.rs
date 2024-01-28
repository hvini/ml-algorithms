pub struct LinearRegression {
    pub weights: Vec<f32>,
    pub epoch: i32,
    pub learning_rate: f32,
    pub cost_val: f32,
    pub tolerance: f32,
}

impl LinearRegression {
    pub fn new() -> LinearRegression {
        LinearRegression {
            weights: vec![0.0, 0.0],
            epoch: 1500,
            learning_rate: 0.01,
            cost_val: 0.0,
            tolerance: 0.0001,
        }
    }

    pub fn fit(&mut self, x: &Vec<Vec<f32>>, y: &Vec<f32>) {
        for i in 0..self.epoch {
            let mut predicted: Vec<f32> = Vec::new();
            for x in x.iter() {
                let hypothesis: f32 = x[0] * self.weights[0] + x[1] * self.weights[1];
                predicted.push(hypothesis);
            }

            let prev_cost = self.cost_val;

            self.minimize(&x, &y, &predicted);

            if i > 0 && (self.cost_val - prev_cost).abs() < self.tolerance {
                println!("Converged at epoch {}", i);
                break;
            }
        }
    }

    fn cost(&mut self, errors: Vec<f32>, m: i32) -> f32 {
        let mut sum: f32 = 0.0;
        let scaling_factor: f32 = 1.0 / (2.0 * m as f32);

        for error in errors.iter() {
            sum += error.powf(2.0);
        }

        return scaling_factor * sum;
    }

    fn minimize(&mut self, x: &Vec<Vec<f32>>, y: &Vec<f32>, predicted: &Vec<f32>) {
        let mut sum_slope: f32 = 0.0;
        let mut sum_intercept: f32 = 0.0;
        let m: i32 = x.len() as i32;
        let scaling_factor: f32 = 1.0 / (m as f32);
        let mut errors: Vec<f32> = Vec::new();

        for (i, y_value) in y.iter().enumerate() {
            let error: f32 = predicted[i] - y_value;
            errors.push(error);
            sum_intercept += error;
            sum_slope += error * x[i][1];
        }

        self.cost_val = self.cost(errors, m);
        self.weights[0] -= self.learning_rate * scaling_factor * sum_intercept;
        self.weights[1] -= self.learning_rate * scaling_factor * sum_slope;
    }
}
