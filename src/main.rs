use std::error::Error;
use std::fs::File;
use std::path::Path;
use std::process;
mod linear_regression;
use linear_regression::LinearRegression;

#[derive(Debug, serde::Deserialize)]
struct Record {
    tamanho_populacao: f32,
    orcamento: f32,
}

fn read_csv<P: AsRef<Path>>(filename: P) -> Result<Vec<Record>, Box<dyn Error>> {
    let file = File::open(filename)?;
    let mut rdr = csv::Reader::from_reader(file);
    let records: Vec<Record> = rdr.deserialize().collect::<Result<_, csv::Error>>()?;
    Ok(records)
}

fn main() {
    let filename = "./dados.txt";
    let records = match read_csv(filename) {
        Ok(records) => records,
        Err(err) => {
            println!("error reading CSV file: {}", err);
            process::exit(1);
        }
    };

    let x_values: Vec<Vec<f32>> = records
        .iter()
        .map(|record| vec![1.0, record.tamanho_populacao])
        .collect();
    let y_values: Vec<f32> = records.iter().map(|record| record.orcamento).collect();

    let mut reg = LinearRegression {
        ..LinearRegression::new()
    };
    reg.fit(&x_values, &y_values);
    println!(
        "cost: {}, intercept: {}, slope: {}",
        reg.cost_val, reg.weights[0], reg.weights[1]
    )
}
